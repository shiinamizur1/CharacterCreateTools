"""UIに依存しない処理（パーツ分割、MToonの色、テクスチャの色調整）。"""
import re

import bpy
import numpy as np

# VRoidのマテリアル名 "N00_001_02_Bottoms_01_CLOTH (Instance)" → "Bottoms"
_CATEGORY_RE = re.compile(r"_([A-Za-z]+)_\d+_(?:SKIN|CLOTH|HAIR|FACE|EYE|ACCE\w*)", re.IGNORECASE)

ORIG_IMAGE_KEY = "vct_orig_images"   # マテリアルに保存する「元のテクスチャ名」
ADJUSTED_SUFFIX = "_vct"


# ---- アーマチュアとパーツ ----------------------------------------------------
DONOR_KEY = "vct_donor"   # 差し替え用に読み込んだVRM（ドナー）の印


def is_vrm_armature(obj):
    return obj is not None and obj.type == "ARMATURE" and hasattr(obj.data, "vrm_addon_extension")


def is_target_armature(obj):
    return is_vrm_armature(obj) and DONOR_KEY not in obj


def find_armature(context):
    obj = context.active_object
    if obj is not None and obj.type != "ARMATURE" and obj.parent is not None:
        obj = obj.parent
    if is_target_armature(obj):
        return obj
    for o in context.scene.objects:
        if is_target_armature(o):
            return o
    return None


def part_objects(armature):
    return sorted((o for o in armature.children if o.type == "MESH"), key=lambda o: o.name)


def material_category(material_name):
    m = _CATEGORY_RE.search(material_name)
    if m:
        return m.group(1)
    return material_name.split(" (")[0]


def _has_shape_keys(obj):
    keys = obj.data.shape_keys
    return keys is not None and len(keys.key_blocks) > 1


def _main_material(obj):
    counts = {}
    for poly in obj.data.polygons:
        counts[poly.material_index] = counts.get(poly.material_index, 0) + 1
    if not counts or not obj.data.materials:
        return None
    return obj.data.materials[max(counts, key=counts.get)]


def _select_only(context, objs, active):
    for o in context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    context.view_layer.objects.active = active


def split_parts(context, armature):
    """マテリアルの種類（VRoidの命名）ごとにメッシュを分け、パーツ名を付ける。

    シェイプキー（表情）を持つメッシュは、表情の設定が壊れないようにそのまま残す。
    戻り値：作られた/確認されたパーツ名のリスト
    """
    if context.object is not None and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    parts = []
    for obj in part_objects(armature):
        cats = {material_category(m.name) for m in obj.data.materials if m is not None}
        if _has_shape_keys(obj) or len(cats) <= 1:
            if "vct_part" not in obj:
                obj["vct_part"] = next(iter(cats)) if len(cats) == 1 and not _has_shape_keys(obj) else obj.name
            parts.append(obj["vct_part"])
            continue

        _select_only(context, [obj], obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="MATERIAL")
        bpy.ops.object.mode_set(mode="OBJECT")
        pieces = list(context.selected_objects)

        groups = {}
        for piece in pieces:
            mat = _main_material(piece)
            groups.setdefault(material_category(mat.name) if mat else piece.name, []).append(piece)

        for cat, objs in groups.items():
            _select_only(context, objs, objs[0])
            if len(objs) > 1:
                bpy.ops.object.join()
            part = context.view_layer.objects.active
            bpy.ops.object.material_slot_remove_unused()
            part.name = cat
            part.data.name = cat
            part["vct_part"] = cat
            parts.append(cat)
    return parts


# ---- MToonの色 ---------------------------------------------------------------
def mtoon(material):
    ext = getattr(material, "vrm_addon_extension", None)
    if ext is None or not ext.mtoon1.enabled:
        return None
    return ext.mtoon1


def texture_slots(material):
    """色調整の対象になるテクスチャ設定（ベースカラー、影色）。"""
    mt = mtoon(material)
    if mt is None:
        return []
    return [
        ("base", mt.pbr_metallic_roughness.base_color_texture.index),
        ("shade", mt.extensions.vrmc_materials_mtoon.shade_multiply_texture.index),
    ]


def _rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h /= 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    choices = [
        np.stack([v, t, p], 1), np.stack([q, v, p], 1), np.stack([p, v, t], 1),
        np.stack([p, q, v], 1), np.stack([t, p, v], 1), np.stack([v, p, q], 1),
    ]
    out = np.empty((len(h), 3), dtype=np.float32)
    for k in range(6):
        m = i == k
        out[m] = choices[k][m]
    return out


def adjusted_image(src, hue, saturation, value):
    """srcの色相・彩度・明度を変えた画像を作る（同じ名前があれば上書き）。"""
    w, h = src.size
    px = np.empty(w * h * 4, dtype=np.float32)
    src.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    hh, ss, vv = _rgb_to_hsv(px[:, :3])
    hh = (hh + hue) % 1.0
    ss = np.clip(ss * saturation, 0.0, 1.0)
    vv = np.clip(vv * value, 0.0, 1.0)
    px[:, :3] = _hsv_to_rgb(hh, ss, vv)

    name = src.name + ADJUSTED_SUFFIX
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


def apply_color_adjust(material, hue, saturation, value):
    """マテリアルのテクスチャに色調整をかける。何度かけても元の画像から計算する。"""
    originals = dict(material.get(ORIG_IMAGE_KEY, {}))
    done = {}
    for key, tex in texture_slots(material):
        if tex.source is None:
            continue
        orig_name = originals.get(key, tex.source.name)
        src = bpy.data.images.get(orig_name)
        if src is None:
            continue
        originals[key] = orig_name
        src.use_fake_user = True   # 使われなくなっても保存時に消えないようにする
        if orig_name not in done:
            done[orig_name] = adjusted_image(src, hue, saturation, value)
        tex.source = done[orig_name]
    material[ORIG_IMAGE_KEY] = originals
    return len(done)


def reset_color_adjust(material):
    originals = dict(material.get(ORIG_IMAGE_KEY, {}))
    for key, tex in texture_slots(material):
        src = bpy.data.images.get(originals.get(key, ""))
        if src is not None:
            tex.source = src
    if ORIG_IMAGE_KEY in material:
        del material[ORIG_IMAGE_KEY]
