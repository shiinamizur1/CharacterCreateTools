"""表情：顔の表情用シェイプキー（VRoidの Fcl_*）を組み合わせて、VRM 1.0 の表情として保存する。

  - ミキサー：顔のシェイプキーの値を直接いじって表情を作る（その場で見た目が変わる）
  - 保存：値が入っているシェイプキーを、VRMの表情（プリセット / カスタム）の
          モーフターゲットのバインドとして書き込む
  - 読み込み：既存の表情のバインドをミキサーに読み込んで編集する

VRMアドオンは表情の設定を書き換えるたびにプレビュー値からシェイプキーを設定し直すので、
書き込みの前後でミキサーの値を控えて戻す。
"""
import bpy

from . import shape

# シェイプキー名の頭 → 表示するグループ名
GROUPS = [
    ("Fcl_ALL_", "全体"),
    ("Fcl_BRW_", "眉"),
    ("Fcl_EYE_", "目"),
    ("Fcl_MTH_", "口"),
    ("Fcl_HA_", "歯"),
]
OTHER_GROUP = "その他"


def expressions(armature):
    return armature.data.vrm_addon_extension.vrm1.expressions


def all_expressions(armature):
    """(表示名, 表情, カスタムかどうか) のリスト。"""
    exprs = expressions(armature)
    out = [(name, e, False) for name, e in exprs.preset.name_to_expression_dict().items()]
    out += [(e.custom_name, e, True) for e in exprs.custom]
    return out


def find_expression(armature, name):
    return expressions(armature).all_name_to_expression_dict().get(name)


def mixer_keys(face):
    """ミキサーに出すシェイプキー（基本形と顔の形スライダーは除く）。"""
    keys = face.data.shape_keys
    if keys is None:
        return []
    ref = keys.reference_key
    return [kb for kb in keys.key_blocks
            if kb != ref and not kb.name.startswith(shape.FACE_KEY_PREFIX)]


def grouped_mixer_keys(face):
    groups = {label: [] for _p, label in GROUPS}
    groups[OTHER_GROUP] = []
    for kb in mixer_keys(face):
        label = next((lab for p, lab in GROUPS if kb.name.startswith(p)), OTHER_GROUP)
        groups[label].append(kb)
    return [(label, kbs) for label, kbs in groups.items() if kbs]


def _mix(face):
    return {kb.name: kb.value for kb in mixer_keys(face)}


def _restore_mix(face, mix):
    keys = face.data.shape_keys.key_blocks
    for name, v in mix.items():
        if name in keys:
            keys[name].value = v


def clear_previews(armature):
    for _name, e, _custom in all_expressions(armature):
        if e.preview > 0.0:
            e.preview = 0.0


def reset_mix(armature, face):
    clear_previews(armature)
    for kb in mixer_keys(face):
        kb.value = 0.0


def load_to_mix(armature, face, name):
    """表情のバインドをミキサーに読み込む（このメッシュの分だけ）。"""
    e = find_expression(armature, name)
    if e is None:
        return 0
    reset_mix(armature, face)
    keys = face.data.shape_keys.key_blocks
    n = 0
    for bind in e.morph_target_binds:
        if bind.node.mesh_object_name == face.name and bind.index in keys:
            keys[bind.index].value = bind.weight
            n += 1
    return n


def save_mix(armature, face, name, create_custom):
    """今のミキサーの値を表情として保存する。create_custom なら新しいカスタム表情を作る。
    戻り値：保存した表情の名前（カスタムで同名があれば .001 などが付く）"""
    mix = _mix(face)
    clear_previews(armature)
    if create_custom:
        bpy.ops.vrm.add_vrm1_expressions_custom_expression(
            armature_object_name=armature.name, custom_expression_name=name)
        e = expressions(armature).custom[-1]
        name = e.custom_name
    else:
        e = find_expression(armature, name)
        if e is None:
            raise RuntimeError(f"表情 {name} が見つかりません")

    # このメッシュへのバインドだけ入れ替える（他のメッシュへのバインドは残す）
    for i in reversed(range(len(e.morph_target_binds))):
        if e.morph_target_binds[i].node.mesh_object_name == face.name:
            e.morph_target_binds.remove(i)
    for key_name, v in mix.items():
        if v > 0.001:
            bind = e.morph_target_binds.add()
            bind.node.mesh_object_name = face.name
            bind.index = key_name
            bind.weight = min(v, 1.0)
    _restore_mix(face, mix)
    return name


def remove_custom(armature, name):
    clear_previews(armature)
    bpy.ops.vrm.remove_vrm1_expressions_custom_expression(
        armature_object_name=armature.name, custom_expression_name=name)
