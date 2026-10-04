"""パーツの差し替え：別のVRM（ドナー）のパーツを、編集中のキャラ（ターゲット）に付け替える。

付け替えの流れ:
  1. ボーンの対応を決める
     - VRMのヒューマノイドボーン同士は役割で対応させる
     - 同じ名前のボーン（J_Bip_* など）は名前で対応させる
     - 揺れ骨（J_Sec_*）は髪型ごとに別物なので、名前の頭にドナー名を付けてターゲットへコピーする
       （胸の揺れ骨のように体側のものは SHARED_SECONDARY で名前対応させる）
  2. 体型の差を補正する：ドナーの各ボーンをターゲットの同じボーンに重ねる変換を、
     ウェイトで混ぜて頂点を動かす（スキニングと同じ計算）
  3. メッシュをターゲットのアーマチュアに付け替え、頂点グループ名を書き換える
  4. コピーした揺れ骨の揺れ設定（VRM SpringBone）もコピーする
"""
import re

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import core

DONOR_KEY = core.DONOR_KEY
SHARED_SECONDARY = ("Bust",)    # 名前が同じなら体側の骨として共有する揺れ骨
DONOR_OFFSET_X = 1.0            # ドナーを横に並べて置く距離（メートル）
NO_PUSH_KINDS = ("Body", "Face", "Hair", "HairBack")  # 体のはみ出し直しをしないパーツ
PUSH_MARGIN = 0.004             # 服を体の表面からどれだけ離すか（メートル）
PUSH_MAX_DIST = 0.04            # これより体から離れた頂点は動かさない（メートル）


# ---- ドナーの読み込みと片付け ------------------------------------------------
def import_donor(context, filepath, stem):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.vrm(filepath=filepath)
    new_objs = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new_objs if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("VRMのアーマチュアが読み込まれませんでした")
    donor = arms[0]
    tag = re.sub(r"[^A-Za-z0-9]", "", stem) or "Donor"
    donor[DONOR_KEY] = tag
    donor.name = f"Donor_{tag}"
    donor.location.x += DONOR_OFFSET_X * len(donor_armatures(context.scene))
    context.view_layer.update()
    core.split_parts(context, donor)
    return donor


def donor_armatures(scene):
    return [o for o in scene.objects if o.type == "ARMATURE" and DONOR_KEY in o]


def remove_donor(donor):
    objs = []

    def collect(o):
        objs.append(o)
        for c in o.children:
            collect(c)

    collect(donor)
    arm_data = donor.data
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    if arm_data.users == 0:
        bpy.data.armatures.remove(arm_data)


# ---- ボーンの対応 ------------------------------------------------------------
def _human_bone_map(armature):
    """ヒューマノイドの役割名 → ボーン名"""
    hb = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    out = {}
    for role, prop in hb.human_bone_name_to_human_bone().items():
        if prop.node.bone_name:
            out[role] = prop.node.bone_name
    return out


def _is_shared_by_name(name):
    if not name.startswith("J_Sec_"):
        return True
    return any(s in name for s in SHARED_SECONDARY)


def build_bone_map(donor, target, tag):
    """ドナーのボーン名 → (ターゲットのボーン名, コピーするか)"""
    tgt_bones = target.data.bones
    mapping = {}
    d_h, t_h = _human_bone_map(donor), _human_bone_map(target)
    for role, dname in d_h.items():
        if role in t_h:
            mapping[dname] = (t_h[role], False)
    for b in donor.data.bones:
        if b.name in mapping:
            continue
        if b.name in tgt_bones and _is_shared_by_name(b.name):
            mapping[b.name] = (b.name, False)
        else:
            mapping[b.name] = (f"{tag}_{b.name}", True)
    return mapping


# ---- 体型差の補正 ------------------------------------------------------------
def _segment_length(bone, mapped_names):
    """ボーンの付け根から、対応のある子ボーンの付け根までの距離（なければNone）。"""
    for c in bone.children:
        if c.name in mapped_names:
            return (c.head_local - bone.head_local).length
    return None


def bone_fit_matrices(donor, target, mapping):
    """ドナーの各ボーンについて、ワールド座標でドナー→ターゲットに重ねる行列。"""
    wd, wt = donor.matrix_world, target.matrix_world
    shared = {d for d, (t, copy) in mapping.items() if not copy}
    mats = {}

    def visit(bone, parent_mat):
        tname, copy = mapping[bone.name]
        if not copy and tname in target.data.bones:
            tb = target.data.bones[tname]
            dl = _segment_length(bone, shared)
            tchild = None
            for c in bone.children:
                if c.name in shared and mapping[c.name][0] in target.data.bones:
                    tchild = target.data.bones[mapping[c.name][0]]
                    break
            scale = 1.0
            if dl and tchild is not None and dl > 1e-5:
                scale = (tchild.head_local - tb.head_local).length / dl
            s = Matrix.Diagonal((1.0, scale, 1.0, 1.0))
            m = wt @ tb.matrix_local @ s @ (wd @ bone.matrix_local).inverted()
        else:
            m = parent_mat
        mats[bone.name] = m
        for c in bone.children:
            visit(c, m)

    for root in (b for b in donor.data.bones if b.parent is None):
        visit(root, wt @ wd.inverted())
    return mats


def _deform_coords(coords_world, weights, group_mats, fallback):
    """coords_world: (N,3) / weights: (N,G) / group_mats: (G,4,4)"""
    n = len(coords_world)
    homo = np.hstack([coords_world, np.ones((n, 1))])
    total = weights.sum(axis=1)
    out = np.einsum("gij,nj->ngi", group_mats, homo)          # (N,G,4)
    blended = np.einsum("ng,ngi->ni", weights, out)
    safe = np.where(total > 1e-8, total, 1.0)[:, None]
    blended = blended / safe
    fb = homo @ np.array(fallback).T
    blended[total <= 1e-8] = fb[total <= 1e-8]
    return blended[:, :3]


def fit_mesh(obj, mats, fallback):
    """メッシュ（とシェイプキー）を、ボーンごとの行列で変形してターゲットの体型に合わせる。"""
    me = obj.data
    groups = obj.vertex_groups
    g_mats = np.array([np.array(mats.get(g.name, fallback)) for g in groups]) if len(groups) else np.zeros((0, 4, 4))
    weights = np.zeros((len(me.vertices), max(len(groups), 1)))
    for v in me.vertices:
        for ge in v.groups:
            if ge.group < len(groups):
                weights[v.index, ge.group] = ge.weight
    if not len(groups):
        g_mats = np.array([np.array(fallback)])

    mw = np.array(obj.matrix_world)
    mw_inv = np.linalg.inv(mw)

    def transform(flat):
        co = flat.reshape(-1, 3)
        world = (np.hstack([co, np.ones((len(co), 1))]) @ mw.T)[:, :3]
        fitted = _deform_coords(world, weights, g_mats, fallback)
        local = (np.hstack([fitted, np.ones((len(fitted), 1))]) @ mw_inv.T)[:, :3]
        return local.ravel()

    co = np.empty(len(me.vertices) * 3)
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            kb.data.foreach_get("co", co)
            kb.data.foreach_set("co", transform(co))
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_set("co", transform(co))
    me.update()


# ---- ボーンと揺れ設定のコピー ------------------------------------------------
def copy_bones(context, donor, target, mapping, mats, needed):
    """needed（ドナーのボーン名）のうちコピー対象のボーンを、親をたどってターゲットに作る。"""
    to_copy = []
    for name in needed:
        b = donor.data.bones.get(name)
        while b is not None:
            tname, copy = mapping[b.name]
            if copy and b.name not in to_copy:
                to_copy.append(b.name)
            b = b.parent
    if not to_copy:
        return []
    # 親から順に作る
    depth = {n: len(donor.data.bones[n].parent_recursive) for n in to_copy}
    to_copy.sort(key=lambda n: depth[n])

    wt_inv = target.matrix_world.inverted()
    wd = donor.matrix_world
    prev_active = context.view_layer.objects.active
    core._select_only(context, [target], target)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = target.data.edit_bones
    created = []
    for name in to_copy:
        b = donor.data.bones[name]
        tname = mapping[name][0]
        m = wt_inv @ mats[name] @ wd
        eb = ebs.get(tname)
        if eb is None:
            eb = ebs.new(tname)
            created.append(tname)
        eb.head = m @ b.head_local
        eb.tail = m @ b.tail_local
        z = (m.to_3x3() @ b.matrix_local.to_3x3().col[2]).normalized()
        eb.align_roll(z)
        eb.use_deform = b.use_deform
        if b.parent is not None:
            eb.parent = ebs.get(mapping[b.parent.name][0])
    bpy.ops.object.mode_set(mode="OBJECT")
    if prev_active is not None:
        context.view_layer.objects.active = prev_active
    return created


def copy_springs(donor, target, mapping, copied_targets, tag):
    """コピーしたボーンを含むドナーの揺れ設定を、ボーン名を書き換えてターゲットに追加する。"""
    d_sb = donor.data.vrm_addon_extension.spring_bone1
    t_sb = target.data.vrm_addon_extension.spring_bone1
    copied = set(copied_targets)
    t_groups = {g.vrm_name: g.uuid for g in t_sb.collider_groups}
    d_groups = {g.uuid: g.vrm_name for g in d_sb.collider_groups}
    existing = {s.vrm_name for s in t_sb.springs}
    added = 0
    for spring in d_sb.springs:
        names = [mapping.get(j.node.bone_name, (j.node.bone_name, False))[0] for j in spring.joints]
        if not names or not any(n in copied for n in names):
            continue
        vrm_name = f"{tag}_{spring.vrm_name}"
        if vrm_name in existing:
            continue
        new = t_sb.add_spring()
        new.vrm_name = vrm_name
        for j, n in zip(spring.joints, names):
            nj = new.joints.add()
            nj.node.bone_name = n
            nj.hit_radius = j.hit_radius
            nj.stiffness = j.stiffness
            nj.gravity_power = j.gravity_power
            nj.gravity_dir = j.gravity_dir
            nj.drag_force = j.drag_force
        for ref in spring.collider_groups:
            uid = t_groups.get(d_groups.get(ref.collider_group_uuid, ""))
            if uid:
                new.collider_groups.add().collider_group_uuid = uid
        if spring.center.bone_name:
            new.center.bone_name = mapping.get(spring.center.bone_name, (spring.center.bone_name, False))[0]
        added += 1
    return added


# ---- 付け替え本体 ------------------------------------------------------------
def attach_part(context, donor_part, target, hide_same_kind=True, push_out=True):
    context.view_layer.update()
    donor = donor_part.parent
    tag = donor[DONOR_KEY]
    mapping = build_bone_map(donor, target, tag)
    mats = bone_fit_matrices(donor, target, mapping)
    fallback = target.matrix_world @ donor.matrix_world.inverted()

    part = donor_part.copy()
    part.data = donor_part.data.copy()
    for col in target.users_collection:
        col.objects.link(part)
        break

    # VRoidのメッシュは全ボーン分の頂点グループを持つので、重みのあるものだけを対象にする
    weighted = set()
    for v in part.data.vertices:
        for ge in v.groups:
            if ge.weight > 0.0:
                weighted.add(part.vertex_groups[ge.group].name)
    for g in list(part.vertex_groups):
        if g.name not in weighted and mapping.get(g.name, ("", True))[1]:
            part.vertex_groups.remove(g)
    used = [n for n in weighted if n in mapping]

    fit_mesh(part, mats, fallback)
    copied = copy_bones(context, donor, target, mapping, mats, used)
    for g in part.vertex_groups:
        if g.name in mapping:
            g.name = mapping[g.name][0]

    # fit_meshの結果はワールド座標で合わせてあるので、それをメッシュに焼き込んでから付け替える
    part.data.transform(part.matrix_world)
    part.parent = target
    part.matrix_parent_inverse = target.matrix_world.inverted()
    part.matrix_basis = Matrix.Identity(4)
    for mod in part.modifiers:
        if mod.type == "ARMATURE":
            mod.object = target

    kind = donor_part.get("vct_part", donor_part.name)
    if hide_same_kind:
        for o in core.part_objects(target):
            if o is not part and o.get("vct_part") == kind:
                o.hide_viewport = True
    part.name = f"{kind}_{tag}"
    part.data.name = part.name
    part["vct_part"] = kind
    part["vct_source"] = tag
    springs = copy_springs(donor, target, mapping, copied, tag)
    body = find_body(target)
    if push_out and kind not in NO_PUSH_KINDS and body is not None:
        push_out_of_body(context, part, body)
    return part, len(copied), springs


# ---- 体のはみ出し直し --------------------------------------------------------
def find_body(target):
    for o in core.part_objects(target):
        if o.get("vct_part") == "Body" and not o.hide_viewport:
            return o
    return None


def push_out_of_body(context, cloth, body, margin=PUSH_MARGIN, max_dist=PUSH_MAX_DIST):
    """体が服を突き抜けている所で、服の頂点を体の表面の外側へ押し出す。動かした頂点数を返す。"""
    depsgraph = context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body, depsgraph)
    to_body = body.matrix_world.inverted() @ cloth.matrix_world
    to_cloth = to_body.inverted()
    rot = to_body.to_3x3()

    me = cloth.data
    deltas = {}
    for v in me.vertices:
        p = to_body @ v.co
        loc, normal, _, dist = tree.find_nearest(p, max_dist)
        if loc is None:
            continue
        depth = (p - loc).dot(normal)
        if depth < margin:
            target = loc + normal * margin
            deltas[v.index] = to_cloth @ target - v.co
    if not deltas:
        return 0
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            for i, d in deltas.items():
                kb.data[i].co += d
    for i, d in deltas.items():
        me.vertices[i].co += d
    me.update()
    return len(deltas)


# ---- パーツごとの書き出し ----------------------------------------------------
def export_parts(context, target, directory, fmt):
    """表示中のパーツを1つずつ、アーマチュア全体と一緒に書き出す。"""
    import os

    parts = [o for o in core.part_objects(target) if not o.hide_viewport]
    saved = {o: o.hide_viewport for o in core.part_objects(target)}
    written = []
    try:
        for part in parts:
            for o in saved:
                o.hide_viewport = o is not part
            base = f"{target.name}_{part.name}"
            if fmt == "VRM":
                path = os.path.join(directory, base + ".vrm")
                bpy.ops.export_scene.vrm(filepath=path, armature_object_name=target.name,
                                         ignore_warning=True)
            else:
                path = os.path.join(directory, base + ".fbx")
                core._select_only(context, [target, part], target)
                bpy.ops.export_scene.fbx(filepath=path, use_selection=True,
                                         object_types={"ARMATURE", "MESH"},
                                         add_leaf_bones=False, bake_anim=False,
                                         mesh_smooth_type="FACE")
            written.append(path)
    finally:
        for o, h in saved.items():
            o.hide_viewport = h
    return written
