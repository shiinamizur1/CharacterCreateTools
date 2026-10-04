"""体型と顔の形。

体型：
  骨ごとに「スケール」を決めて新しい骨の配置を計算し、メッシュは swap.fit_mesh と同じ
  スキニング計算で「今の配置 → 新しい配置」に変形する。最初に適用したときの骨の配置を
  基準（vct_base）として保存し、スライダーの値は常にその基準からの倍率として扱う。
  - subtree：その骨から先をまるごと均等に拡大（身長・頭・手・足・胸）
  - segment：その骨の区間だけを軸方向に伸ばす（首・胴・腕・脚の長さ、太さ、肩幅）

顔の形：
  顔のマテリアル（目・眉・口・肌）から目や口の位置を求め、形を変えるシェイプキー
  （VCT_*）を自動で作る。値はスライダーでその場で変えられ、UEではモーフターゲットになる。
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import core, swap

BASE_KEY = "vct_base"          # 基準の骨配置（アーマチュアのデータに保存）
APPLIED_KEY = "vct_applied"    # 今メッシュに適用されている体型の値

# (プロパティ名, 表示名, 種類, 対象（役割名 / "#root" / "#bust"）, 軸)
BODY_PARAMS = [
    ("height", "身長", "subtree", ["#root"], ""),
    ("head", "頭の大きさ", "subtree", ["head"], ""),
    ("neck", "首の長さ", "segment", ["neck"], "Z"),
    ("shoulder", "肩幅", "segment", ["leftShoulder", "rightShoulder"], "X"),
    ("torso", "胴の長さ", "segment", ["spine", "chest", "upperChest"], "Z"),
    ("body_width", "胴の太さ", "segment", ["hips", "spine", "chest", "upperChest"], "XY"),
    ("bust", "胸の大きさ", "subtree", ["#bust"], ""),
    ("arm_length", "腕の長さ", "segment",
     ["leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm"], "X"),
    ("arm_thick", "腕の太さ", "segment",
     ["leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm"], "YZ"),
    ("hand", "手の大きさ", "subtree", ["leftHand", "rightHand"], ""),
    ("leg_length", "脚の長さ", "segment",
     ["leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg"], "Z"),
    ("leg_thick", "脚の太さ", "segment",
     ["leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg"], "XY"),
    ("foot", "足の大きさ", "subtree", ["leftFoot", "rightFoot"], ""),
]
_AXIS = {"X": 0, "Y": 1, "Z": 2}


# ---- 体型 --------------------------------------------------------------------
def _role_bones(armature):
    out = {}
    for role, name in swap._human_bone_map(armature).items():
        out[str(getattr(role, "value", role))] = name
    return out


def _factor_tables(armature, values):
    """骨ごとの subtree 倍率と segment の軸倍率を作る。"""
    roles = _role_bones(armature)
    bones = armature.data.bones
    sub = {}   # name -> float
    seg = {}   # name -> [sx, sy, sz]
    for prop, _label, kind, targets, axes in BODY_PARAMS:
        v = values.get(prop, 1.0)
        if abs(v - 1.0) < 1e-6:
            continue
        names = []
        for t in targets:
            if t == "#root":
                names += [b.name for b in bones if b.parent is None]
            elif t == "#bust":
                names += [b.name for b in bones
                          if "Bust" in b.name and (b.parent is None or "Bust" not in b.parent.name)]
            elif t in roles:
                names.append(roles[t])
        for n in names:
            if kind == "subtree":
                sub[n] = sub.get(n, 1.0) * v
            else:
                s = seg.setdefault(n, [1.0, 1.0, 1.0])
                for a in axes:
                    s[_AXIS[a]] *= v
    return sub, seg, roles


def _ensure_base(armature):
    """基準の骨配置がなければ、今の配置を基準として保存する。"""
    data = armature.data
    if BASE_KEY in data:
        return
    base = {}
    for b in data.bones:
        z = b.matrix_local.to_3x3().col[2]
        base[b.name] = list(b.head_local) + list(b.tail_local) + list(z)
    data[BASE_KEY] = base
    data[APPLIED_KEY] = {p[0]: 1.0 for p in BODY_PARAMS}


def _layout(armature, base, values):
    """基準の配置と体型の値から、骨ごとの (新しい付け根, 3x3スケール, 基準の付け根) を計算する。"""
    sub, seg, roles = _factor_tables(armature, values)
    out = {}

    def visit(bone, parent):
        bb = base[bone.name]
        bh = Vector(bb[0:3])
        c_parent = out[parent.name][3] if parent else 1.0
        c = c_parent * sub.get(bone.name, 1.0)
        if parent is None:
            head = bh * c
        else:
            ph, ps, pbh, _ = out[parent.name]
            head = ph + ps @ (bh - pbh)
        s = Matrix.Diagonal(seg.get(bone.name, [1.0, 1.0, 1.0])) * c
        out[bone.name] = (head, s, bh, c)
        for ch in bone.children:
            visit(ch, bone)

    for root in (b for b in armature.data.bones if b.parent is None):
        visit(root, None)

    # 脚の長さが変わっても足が地面に着いたままになるよう、全体を上下にずらす
    foot = roles.get("leftFoot")
    if foot in out:
        head, _s, bh, _c = out[foot]
        root_c = values.get("height", 1.0)
        dz = bh.z * root_c - head.z
        for n, (h, s, b, c) in out.items():
            out[n] = (h + Vector((0, 0, dz)), s, b, c)
    return out


def _bone_matrices(layout):
    """骨ごとに 基準の位置 → 新しい位置 の4x4行列（アーマチュア空間）。"""
    mats = {}
    for n, (head, s, bh, _c) in layout.items():
        mats[n] = Matrix.Translation(head) @ s.to_4x4() @ Matrix.Translation(-bh)
    return mats


def _add_missing_base(armature, base, current_mats):
    """基準を保存した後で追加された骨（付けたパーツの揺れ骨など）の基準位置を、
    親の今の変形を逆にかけて求める。"""
    added = False
    for b in armature.data.bones:
        if b.name in base:
            continue
        p = b.parent
        while p is not None and p.name not in current_mats:
            p = p.parent
        inv = current_mats[p.name].inverted() if p is not None else Matrix.Identity(4)
        z = b.matrix_local.to_3x3().col[2]
        base[b.name] = (list(inv @ b.head_local) + list(inv @ b.tail_local)
                        + list((inv.to_3x3() @ z).normalized()))
        added = True
    return added


def body_values(armature):
    return {p[0]: getattr(armature.data.vct_body, p[0]) for p in BODY_PARAMS}


def apply_body_shape(context, armature, values):
    """体型の値をメッシュと骨に適用する。"""
    _ensure_base(armature)
    data = armature.data
    base = {k: list(v) for k, v in data[BASE_KEY].items()}
    applied = dict(data[APPLIED_KEY])

    # 基準にない骨は、今の変形から逆算して基準に加える（親から順に）
    for _ in range(8):
        cur_mats = _bone_matrices(_layout_partial(armature, base, applied))
        if not _add_missing_base(armature, base, cur_mats):
            break
    data[BASE_KEY] = base

    cur = _bone_matrices(_layout(armature, base, applied))
    new_layout = _layout(armature, base, values)
    new = _bone_matrices(new_layout)

    w = armature.matrix_world
    w_inv = w.inverted()
    world = {n: w @ new[n] @ cur[n].inverted() @ w_inv for n in new}
    for obj in core.part_objects(armature):
        swap.fit_mesh(obj, world, Matrix.Identity(4))

    core._select_only(context, [armature], armature)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = data.edit_bones
    for eb in ebs:
        if eb.name not in new_layout:
            continue
        head, s, bh, _c = new_layout[eb.name]
        bb = base[eb.name]
        eb.head = head
        eb.tail = head + s @ (Vector(bb[3:6]) - bh)
        eb.align_roll((s @ Vector(bb[6:9])).normalized())
    bpy.ops.object.mode_set(mode="OBJECT")
    data[APPLIED_KEY] = dict(values)


def _layout_partial(armature, base, values):
    """基準にまだない骨を飛ばして配置を計算する（_add_missing_base 用）。"""
    sub, seg, _roles = _factor_tables(armature, values)
    out = {}

    def visit(bone, parent_entry):
        if bone.name not in base:
            return
        bb = base[bone.name]
        bh = Vector(bb[0:3])
        if parent_entry is None:
            c = sub.get(bone.name, 1.0)
            head = bh * c
        else:
            ph, ps, pbh, pc = parent_entry
            c = pc * sub.get(bone.name, 1.0)
            head = ph + ps @ (bh - pbh)
        s = Matrix.Diagonal(seg.get(bone.name, [1.0, 1.0, 1.0])) * c
        out[bone.name] = (head, s, bh, c)
        for ch in bone.children:
            visit(ch, out[bone.name])

    for root in (b for b in armature.data.bones if b.parent is None):
        visit(root, None)
    return out


# ---- 顔の形 ------------------------------------------------------------------
FACE_KEY_PREFIX = "VCT_"
# (シェイプキー名, 表示名)
FACE_KEYS = [
    ("VCT_EyeSize", "目の大きさ"),
    ("VCT_EyeHeight", "目の高さ"),
    ("VCT_EyeSpacing", "目の間隔"),
    ("VCT_EyeTilt", "つり目 / たれ目"),
    ("VCT_BrowHeight", "眉の高さ"),
    ("VCT_MouthHeight", "口の高さ"),
    ("VCT_MouthWidth", "口の幅"),
    ("VCT_FaceWidth", "顔の幅"),
    ("VCT_JawSlim", "あごの細さ"),
    ("VCT_ChinLength", "あごの長さ"),
]
EYE_PARTS = ("EyeIris", "EyeWhite", "EyeHighlight", "FaceEyeline")


def find_face(armature):
    for o in core.part_objects(armature):
        keys = o.data.shape_keys
        if keys and any(k.name.startswith("Fcl_") for k in keys.key_blocks):
            return o
    for o in core.part_objects(armature):
        if o.get("vct_part") == "Face":
            return o
    return None


def _vertex_categories(me):
    cats = [set() for _ in me.vertices]
    names = [core.material_category(m.name) if m else "" for m in me.materials]
    for poly in me.polygons:
        c = names[poly.material_index] if poly.material_index < len(names) else ""
        for vi in poly.vertices:
            cats[vi].add(c)
    return cats


def _landmarks(me, co):
    cats = _vertex_categories(me)
    idx = lambda *names: np.array([i for i, c in enumerate(cats) if c & set(names)], dtype=np.int64)
    eye = idx(*EYE_PARTS)
    iris = idx("EyeIris")
    brow = idx("FaceBrow")
    mouth = idx("FaceMouth")
    skin = idx("Face")
    if len(iris) == 0 or len(mouth) == 0 or len(skin) == 0:
        raise RuntimeError("目・口・肌のマテリアルが見つかりません（VRoidの顔ではない可能性があります）")
    eye_c = {s: co[iris[np.sign(co[iris, 0]) == s]].mean(axis=0) for s in (1, -1)}
    mouth_c = co[mouth].mean(axis=0)
    mouth_half = (co[mouth, 0].max() - co[mouth, 0].min()) / 2
    eye_z = (eye_c[1][2] + eye_c[-1][2]) / 2
    front = skin[(np.abs(co[skin, 0]) < 0.006) & (co[skin, 1] < mouth_c[1] + 0.01)]
    chin_z = co[front, 2].min() if len(front) else co[skin, 2].min()

    # 首とのつなぎ目（メッシュの端で口より下）からの距離で、つなぎ目付近は動かさない
    edge_count = {}
    for e in me.edges:
        edge_count[e.key] = 0
    for poly in me.polygons:
        for k in poly.edge_keys:
            edge_count[k] = edge_count.get(k, 0) + 1
    border = {v for k, n in edge_count.items() if n == 1 for v in k}
    seam = [v for v in border if co[v, 2] < mouth_c[2]]
    tree = KDTree(len(seam))
    for i, v in enumerate(seam):
        tree.insert(co[v], i)
    tree.balance()
    seam_w = np.ones(len(co))
    if seam:
        for i in range(len(co)):
            _p, _j, d = tree.find(co[i])
            seam_w[i] = min(d / 0.025, 1.0)
    return dict(eye=eye, brow=brow, mouth=mouth, eye_c=eye_c, mouth_c=mouth_c,
                mouth_half=mouth_half, eye_z=eye_z, chin_z=chin_z, seam_w=seam_w)


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _face_key_coords(name, co, lm):
    out = co.copy()
    eye = lm["eye"]
    side = np.where(co[eye, 0] >= 0, 1, -1)
    centers = np.array([lm["eye_c"][s] for s in side])
    if name == "VCT_EyeSize":
        d = co[eye] - centers
        out[eye] = centers + d * np.array([1.2, 1.0, 1.2])
    elif name == "VCT_EyeHeight":
        out[eye, 2] += 0.006
    elif name == "VCT_EyeSpacing":
        out[eye, 0] += side * 0.004
    elif name == "VCT_EyeTilt":
        a = np.radians(8.0) * side
        d = co[eye] - centers
        out[eye, 0] = centers[:, 0] + d[:, 0] * np.cos(a) - d[:, 2] * np.sin(a)
        out[eye, 2] = centers[:, 2] + d[:, 0] * np.sin(a) + d[:, 2] * np.cos(a)
    elif name == "VCT_BrowHeight":
        out[lm["brow"], 2] += 0.006
    elif name in ("VCT_MouthHeight", "VCT_MouthWidth"):
        mc, r0 = lm["mouth_c"], lm["mouth_half"]
        front = co[:, 1] < mc[1] + 0.02
        dist = np.hypot(co[:, 0] - mc[0], (co[:, 2] - mc[2]) * 1.5)
        w = (1.0 - _smooth((dist - r0) / r0)) * front
        if name == "VCT_MouthHeight":
            out[:, 2] += 0.005 * w
        else:
            out[:, 0] = mc[0] + (co[:, 0] - mc[0]) * (1.0 + 0.2 * w)
    elif name == "VCT_FaceWidth":
        # 頬から下だけを広げる（頭の上まで広げると髪を突き抜ける）
        wz = 1.0 - _smooth((co[:, 2] - lm["eye_z"]) / 0.03)
        out[:, 0] = co[:, 0] * (1.0 + 0.08 * lm["seam_w"] * wz)
    elif name == "VCT_JawSlim":
        t = _smooth((lm["eye_z"] - co[:, 2]) / (lm["eye_z"] - lm["chin_z"])) ** 1.5
        out[:, 0] = co[:, 0] * (1.0 - 0.15 * t * lm["seam_w"])
    elif name == "VCT_ChinLength":
        mz = lm["mouth_c"][2]
        t = _smooth((mz - co[:, 2]) / (mz - lm["chin_z"]))
        out[:, 2] -= 0.012 * t * lm["seam_w"]
    return out


def create_face_keys(face):
    """顔の形のシェイプキーを作る（既にあれば作り直す。値はそのまま）。"""
    me = face.data
    if me.shape_keys is None:
        face.shape_key_add(name="Basis", from_mix=False)
    basis = me.shape_keys.reference_key
    co = np.empty(len(me.vertices) * 3)
    basis.data.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lm = _landmarks(me, co)
    for name, _label in FACE_KEYS:
        kb = me.shape_keys.key_blocks.get(name)
        if kb is None:
            kb = face.shape_key_add(name=name, from_mix=False)
            kb.value = 0.0
        kb.slider_min = -1.0
        kb.slider_max = 1.0
        kb.data.foreach_set("co", _face_key_coords(name, co, lm).ravel())
    me.update()
    return len(FACE_KEYS)


def bake_face_keys(face):
    """顔の形のシェイプキーの今の値を、基本形（Basis）に焼き込んで値を0に戻す。
    VRMの書き出しは基本形を使うので、形を確定させたいときに使う。"""
    keys = face.data.shape_keys
    if keys is None:
        return 0
    n = len(face.data.vertices) * 3
    basis = keys.reference_key
    b = np.empty(n)
    basis.data.foreach_get("co", b)
    delta = np.zeros(n)
    baked = 0
    for name, _label in FACE_KEYS:
        kb = keys.key_blocks.get(name)
        if kb is None or abs(kb.value) < 1e-6:
            continue
        k = np.empty(n)
        kb.data.foreach_get("co", k)
        delta += (k - b) * kb.value
        baked += 1
    if baked == 0:
        return 0
    # 他のキーとの差（表情など）が変わらないよう、全部のキーに同じ差を足す
    for kb in keys.key_blocks:
        k = np.empty(n)
        kb.data.foreach_get("co", k)
        kb.data.foreach_set("co", k + delta)
    face.data.vertices.foreach_set("co", b + delta)
    for name, _label in FACE_KEYS:
        kb = keys.key_blocks.get(name)
        if kb is not None:
            kb.value = 0.0
    face.data.update()
    return baked
