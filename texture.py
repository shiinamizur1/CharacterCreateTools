"""テクスチャ編集：部分的な色替え（マスク）とエンブレム（デカール）の貼り込み。

マテリアルごとに「元の画像」を残しておき、次の順に毎回作り直して *_vct 画像に書き出す。
  1. 全体の色相・彩度・明度（パーツと色パネルの値）
  2. 部分の色替え：マスクの範囲を、陰影を保ったまま目標の色に寄せる
  3. エンブレム：UV上の位置・大きさ・回転で画像を重ねる

マスクはテクスチャと同じUVの白黒画像。作り方は3通り：
  - 編集モードで選んだ面のUVを塗る
  - 元のテクスチャで似た色の画素を選ぶ
  - テクスチャペイントで手で塗る
"""
import math

import bpy
import numpy as np
from bpy.props import (BoolProperty, FloatProperty, FloatVectorProperty, PointerProperty,
                       StringProperty)
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

from . import core

MASK_PREFIX = "VCT_Mask_"
_pixel_cache = {}   # 元画像の画素（name -> (size, array)）。元画像は変わらないので使い回す


# ---- プロパティ ---------------------------------------------------------------
def _live_update(self, context):
    if context.scene.vct.tex_live:
        rebuild(self.id_data)


def mask_name(material, region_name):
    return f"{MASK_PREFIX}{material.name.replace(' (Instance)', '')}_{region_name}"


def _rename_mask(self, context):
    if self.mask is not None:
        self.mask.name = mask_name(self.id_data, self.name)


class VCT_Region(bpy.types.PropertyGroup):
    name: StringProperty(name="名前", default="部分", update=_rename_mask)
    enabled: BoolProperty(name="有効", default=True, update=_live_update)
    mask: PointerProperty(type=bpy.types.Image, name="マスク")
    color: FloatVectorProperty(name="色", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                               default=(0.9, 0.3, 0.3), update=_live_update,
                               description="この部分をこの色に寄せる（陰影は残る）")
    strength: FloatProperty(name="強さ", min=0.0, max=1.0, default=1.0, update=_live_update)
    pick_color: FloatVectorProperty(name="選ぶ色", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                                    default=(0.0, 0.0, 1.0),
                                    description="スポイトでテクスチャ上の色を拾う")
    tolerance: FloatProperty(name="許容範囲", min=0.01, max=1.0, default=0.15,
                             description="選ぶ色にどれだけ近い色まで含めるか")
    expanded: BoolProperty(default=True)


class VCT_Decal(bpy.types.PropertyGroup):
    name: StringProperty(name="名前", default="エンブレム")
    enabled: BoolProperty(name="有効", default=True, update=_live_update)
    image: PointerProperty(type=bpy.types.Image, name="画像", update=_live_update)
    u: FloatProperty(name="横位置(U)", min=0.0, max=1.0, default=0.5, step=0.1, update=_live_update)
    v: FloatProperty(name="縦位置(V)", min=0.0, max=1.0, default=0.5, step=0.1, update=_live_update)
    scale: FloatProperty(name="大きさ", min=0.005, max=1.0, default=0.1, step=0.1, update=_live_update,
                         description="テクスチャの幅に対するエンブレムの幅")
    rotation: FloatProperty(name="回転", subtype="ANGLE", default=0.0, update=_live_update)
    opacity: FloatProperty(name="不透明度", min=0.0, max=1.0, default=1.0, update=_live_update)
    flip: BoolProperty(name="左右反転", default=False, update=_live_update)
    tint: FloatVectorProperty(name="色（乗算）", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                              default=(1.0, 1.0, 1.0), update=_live_update)
    expanded: BoolProperty(default=True)


PROPERTY_CLASSES = (VCT_Region, VCT_Decal)


# ---- 画素の読み書き -----------------------------------------------------------
def _read(image):
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def _original_pixels(image):
    key = image.name
    cached = _pixel_cache.get(key)
    if cached is None or cached[0] != tuple(image.size):
        cached = (tuple(image.size), _read(image))
        _pixel_cache[key] = cached
    return cached[1].copy()


def _resample(arr, h, w):
    """最近傍で (h, w) に合わせる。"""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    ys = (np.arange(h) * arr.shape[0] / h).astype(np.int64)
    xs = (np.arange(w) * arr.shape[1] / w).astype(np.int64)
    return arr[ys][:, xs]


def mask_values(region, h, w):
    if region.mask is None:
        return None
    m = _read(region.mask)[:, :, 0]
    return _resample(m, h, w)


# ---- 色の計算 -----------------------------------------------------------------
def _global_adjust(px, s):
    if abs(s.hue) < 1e-6 and abs(s.saturation - 1) < 1e-6 and abs(s.value - 1) < 1e-6:
        return
    flat = px.reshape(-1, 4)
    hh, ss, vv = core._rgb_to_hsv(flat[:, :3])
    hh = (hh + s.hue) % 1.0
    ss = np.clip(ss * s.saturation, 0.0, 1.0)
    vv = np.clip(vv * s.value, 0.0, 1.0)
    flat[:, :3] = core._hsv_to_rgb(hh, ss, vv)


def _recolor_region(px, mask, region):
    sel = np.nonzero(mask > 0.004)
    if len(sel[0]) == 0:
        return
    rgb = px[sel][:, :3]
    w = (mask[sel] * region.strength)[:, None]
    hh, ss, vv = core._rgb_to_hsv(rgb)
    weights = mask[sel]
    # 部分の平均色（色相は角度の平均）
    ang = hh * 2 * math.pi
    mh = (math.atan2((np.sin(ang) * weights).sum(), (np.cos(ang) * weights).sum())
          / (2 * math.pi)) % 1.0
    ms = float((ss * weights).sum() / weights.sum())
    mv = float((vv * weights).sum() / weights.sum())
    th, ts, tv = core._rgb_to_hsv(np.array([region.color], dtype=np.float32))
    th, ts, tv = float(th[0]), float(ts[0]), float(tv[0])
    nh = (hh + (th - mh)) % 1.0
    if ms < 0.05:   # 白・灰色の部分は色相を回しても色が付かないので、彩度を直接与える
        ns = np.full_like(ss, ts)
    else:
        ns = np.clip(ss * ts / ms, 0.0, 1.0)
    nv = np.clip(vv * (tv / max(mv, 1e-4)), 0.0, 1.0)
    new = core._hsv_to_rgb(nh, ns, nv)
    px[sel[0], sel[1], :3] = rgb * (1 - w) + new * w


def _bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _draw_decal(px, decal):
    if decal.image is None or decal.image.size[0] == 0:
        return
    H, W = px.shape[:2]
    src = _read(decal.image)
    dh, dw = src.shape[:2]
    cx, cy = decal.u * W, decal.v * H
    hw = decal.scale * W / 2
    hh = hw * dh / dw
    r = math.hypot(hw, hh)
    x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, W)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, H)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx, dy = xs + 0.5 - cx, ys + 0.5 - cy
    c, s = math.cos(-decal.rotation), math.sin(-decal.rotation)
    lx = (dx * c - dy * s) / (2 * hw) + 0.5
    ly = (dx * s + dy * c) / (2 * hh) + 0.5
    if decal.flip:
        lx = 1.0 - lx
    inside = (lx >= 0) & (lx <= 1) & (ly >= 0) & (ly <= 1)
    if not inside.any():
        return
    sample = _bilinear(src, lx[inside] * (dw - 1), ly[inside] * (dh - 1))
    a = (sample[:, 3] * decal.opacity)[:, None]
    tint = np.array(decal.tint, dtype=np.float32)
    region = px[y0:y1, x0:x1]
    base = region[inside][:, :3]
    region[inside, :3] = base * (1 - a) + sample[:, :3] * tint * a


# ---- 作り直し -----------------------------------------------------------------
def has_edits(material):
    s = material.vct
    global_on = not (abs(s.hue) < 1e-6 and abs(s.saturation - 1) < 1e-6 and abs(s.value - 1) < 1e-6)
    return (global_on or any(r.enabled and r.mask for r in s.regions)
            or any(d.enabled and d.image for d in s.decals))


def rebuild(material):
    """マテリアルのテクスチャを元の画像から作り直す。編集がなければ元の画像に戻す。"""
    if not has_edits(material):
        core.reset_color_adjust(material)
        return 0
    s = material.vct
    originals = dict(material.get(core.ORIG_IMAGE_KEY, {}))
    done = {}
    for key, tex in core.texture_slots(material):
        if tex.source is None:
            continue
        orig_name = originals.get(key, tex.source.name)
        src = bpy.data.images.get(orig_name)
        if src is None:
            continue
        originals[key] = orig_name
        src.use_fake_user = True
        if orig_name not in done:
            px = _original_pixels(src)
            h, w = px.shape[:2]
            _global_adjust(px, s)
            for region in s.regions:
                if region.enabled:
                    m = mask_values(region, h, w)
                    if m is not None:
                        _recolor_region(px, m, region)
            for decal in s.decals:
                if decal.enabled:
                    _draw_decal(px, decal)
            done[orig_name] = _write_output(src, px)
        tex.source = done[orig_name]
    material[core.ORIG_IMAGE_KEY] = originals
    return len(done)


def _write_output(src, px):
    h, w = px.shape[:2]
    name = src.name + core.ADJUSTED_SUFFIX
    dst = bpy.data.images.get(name)
    if dst is None or tuple(dst.size) != (w, h):
        if dst is not None:
            bpy.data.images.remove(dst)
        dst = bpy.data.images.new(name, w, h, alpha=True)
    dst.colorspace_settings.name = src.colorspace_settings.name
    dst.alpha_mode = src.alpha_mode
    dst.pixels.foreach_set(px.ravel())
    dst.pack()
    return dst


def original_image(material):
    """マテリアルのベースカラーの元画像。"""
    originals = material.get(core.ORIG_IMAGE_KEY, {})
    for key, tex in core.texture_slots(material):
        if key == "base":
            name = originals.get("base") or (tex.source.name if tex.source else "")
            return bpy.data.images.get(name)
    return None


# ---- マスクを作る -------------------------------------------------------------
def ensure_mask(material, region):
    if region.mask is not None:
        return region.mask
    src = original_image(material)
    w, h = src.size if src is not None else (1024, 1024)
    img = bpy.data.images.new(mask_name(material, region.name), w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    img.generated_color = (0, 0, 0, 1)
    img.pack()
    region.mask = img
    return img


def _write_mask(img, m):
    h, w = m.shape
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[:, :, 0] = px[:, :, 1] = px[:, :, 2] = m
    px[:, :, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()


def _dilate(m, n):
    for _ in range(n):
        g = m.copy()
        g[1:, :] = np.maximum(g[1:, :], m[:-1, :])
        g[:-1, :] = np.maximum(g[:-1, :], m[1:, :])
        g[:, 1:] = np.maximum(g[:, 1:], m[:, :-1])
        g[:, :-1] = np.maximum(g[:, :-1], m[:, 1:])
        m = g
    return m


def mask_from_faces(material, region, objects, mode="ADD"):
    """選択した面（このマテリアルのもの）のUV三角形をマスクに塗る。塗った面の数を返す。"""
    img = ensure_mask(material, region)
    w, h = img.size
    m = np.zeros((h, w), dtype=np.float32) if mode == "REPLACE" else _read(img)[:, :, 0].copy()
    painted = 0
    for obj in objects:
        me = obj.data
        uv_layer = me.uv_layers.active
        if uv_layer is None:
            continue
        mat_index = {i for i, mt in enumerate(me.materials) if mt == material}
        me.calc_loop_triangles()
        for tri in me.loop_triangles:
            poly = me.polygons[tri.polygon_index]
            if not poly.select or tri.material_index not in mat_index:
                continue
            pts = np.array([uv_layer.data[li].uv[:] for li in tri.loops]) * (w, h)
            _fill_triangle(m, pts, 1.0 if mode != "SUBTRACT" else 0.0)
            painted += 1
    m = _dilate(m, 2) if mode != "SUBTRACT" else m
    _write_mask(img, m)
    return painted


def _fill_triangle(m, pts, value):
    h, w = m.shape
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x0 > x1 or y0 > y1:
        return
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    px, py = xs + 0.5, ys + 0.5
    (ax, ay), (bx, by), (cx, cy) = pts
    d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(d) < 1e-9:
        return
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d
    l3 = 1 - l1 - l2
    e = -1e-3   # 継ぎ目の隙間は後で数画素広げて防ぐ
    inside = (l1 >= e) & (l2 >= e) & (l3 >= e)
    m[y0:y1 + 1, x0:x1 + 1][inside] = value


def mask_from_color(material, region, mode="ADD"):
    """元のテクスチャで pick_color に近い色の画素をマスクにする。選ばれた画素数を返す。"""
    src = original_image(material)
    if src is None:
        return 0
    img = ensure_mask(material, region)
    w, h = img.size
    px = _resample(_original_pixels(src), h, w)
    target = np.array(region.pick_color, dtype=np.float32)
    dist = np.linalg.norm(px[:, :, :3] - target, axis=2) / math.sqrt(3)
    hit = (dist <= region.tolerance) & (px[:, :, 3] > 0.01)
    m = _read(img)[:, :, 0].copy()
    if mode == "REPLACE":
        m[:] = 0
        m[hit] = 1.0
    elif mode == "ADD":
        m[hit] = 1.0
    elif mode == "SUBTRACT":
        m[hit] = 0.0
    elif mode == "INTERSECT":
        m = np.where(hit, m, 0.0)
    _write_mask(img, m)
    return int(hit.sum())


def save_masks(material, directory):
    import os
    written = []
    for region in material.vct.regions:
        if region.mask is None:
            continue
        path = os.path.join(directory, region.mask.name + ".png")
        copy = region.mask.copy()
        copy.filepath_raw = path
        copy.file_format = "PNG"
        copy.save()
        bpy.data.images.remove(copy)
        written.append(path)
    return written


# ---- モデルをクリックしてUVを拾う ---------------------------------------------
def pick_uv(context, parts, origin, direction):
    """視線の先で最初に当たるパーツの面から (object, material, uv) を返す（レストポーズ前提）。"""
    best = None
    for obj in parts:
        if obj.hide_viewport or obj.hide_get():
            continue
        me = obj.data
        uv_layer = me.uv_layers.active
        if uv_layer is None:
            continue
        me.calc_loop_triangles()
        mw = obj.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        tree = BVHTree.FromPolygons(verts, tris)
        loc, _n, idx, dist = tree.ray_cast(origin, direction)
        if loc is None or (best is not None and dist >= best[0]):
            continue
        tri = me.loop_triangles[idx]
        a, b, c = (verts[i] for i in tri.vertices)
        uva, uvb, uvc = (Vector((*uv_layer.data[li].uv, 0.0)) for li in tri.loops)
        uv = barycentric_transform(loc, a, b, c, uva, uvb, uvc)
        mat = me.materials[tri.material_index] if tri.material_index < len(me.materials) else None
        best = (dist, obj, mat, (uv.x % 1.0, uv.y % 1.0))
    return best[1:] if best else None
