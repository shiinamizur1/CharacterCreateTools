"""VRM Chara Tool：VRMキャラクターを読み込み、パーツ分け・色編集・パーツ差し替え・体型と顔の形・表情・テクスチャの編集をして書き出す。

3Dビューのサイドバー（Nキー）の「キャラ編集」タブに出る。
VRMの読み書きは VRM Add-on for Blender（vrm拡張機能）を使う。
"""
import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy_extras import view3d_utils
from bpy_extras.io_utils import ImportHelper

from . import core, expr, shape, swap, texture


class VCT_MaterialSettings(bpy.types.PropertyGroup):
    hue: FloatProperty(name="色相", min=-0.5, max=0.5, default=0.0, step=1,
                       description="テクスチャの色相をずらす（-0.5〜0.5で一周）")
    saturation: FloatProperty(name="彩度", min=0.0, max=3.0, default=1.0, step=1,
                              description="テクスチャの彩度の倍率")
    value: FloatProperty(name="明度", min=0.0, max=3.0, default=1.0, step=1,
                         description="テクスチャの明るさの倍率")
    regions: CollectionProperty(type=texture.VCT_Region)
    decals: CollectionProperty(type=texture.VCT_Decal)


def _material_in_target(mat):
    arm = core.find_armature(bpy.context)
    return arm is not None and any(mat in o.data.materials[:] for o in core.part_objects(arm))


class VCT_ObjectSettings(bpy.types.PropertyGroup):
    expanded: BoolProperty(name="展開", default=False)


VCT_BodyShape = type("VCT_BodyShape", (bpy.types.PropertyGroup,), {
    "__annotations__": {
        prop: FloatProperty(name=label, min=0.5, max=1.5, soft_min=0.7, soft_max=1.3,
                            default=1.0, step=1, description=f"{label}の倍率（1.0が元の体型）")
        for prop, label, *_ in shape.BODY_PARAMS
    },
})


class VCT_SceneSettings(bpy.types.PropertyGroup):
    hide_same_kind: BoolProperty(
        name="同じ種類のパーツを隠す", default=True,
        description="付けたパーツと同じ種類（髪なら髪）の今のパーツを非表示にする（非表示は書き出されない）")
    push_out: BoolProperty(
        name="体のはみ出しを直す", default=True,
        description="服を付けたとき、体が服を突き抜けている所で服を外側へ押し出す")
    expr_name: StringProperty(name="名前", default="MyExpression",
                              description="新しく保存する表情の名前")
    expr_editing: StringProperty(name="編集中の表情", default="")
    expr_group: EnumProperty(
        name="グループ",
        items=[(lab, lab, "") for _p, lab in expr.GROUPS] + [(expr.OTHER_GROUP, expr.OTHER_GROUP, "")],
        default="目")
    tex_material: PointerProperty(
        type=bpy.types.Material, name="マテリアル",
        poll=lambda self, mat: _material_in_target(mat),
        description="テクスチャを編集するマテリアル")
    painting: BoolProperty(default=False, options={"SKIP_SAVE"})
    tex_live: BoolProperty(name="すぐ反映", default=True,
                           description="値を変えるたびにテクスチャを作り直す（重いときはオフにして「反映」を押す）")
    mask_mode: EnumProperty(
        name="塗り方",
        items=[("ADD", "追加", "今のマスクに足す"),
               ("REPLACE", "置き換え", "今のマスクを消してから塗る"),
               ("SUBTRACT", "削る", "今のマスクから取り除く"),
               ("INTERSECT", "絞り込み", "今のマスクの中だけに絞る（色で選ぶときのみ）")],
        default="ADD")
    export_format: EnumProperty(
        name="形式",
        items=[("VRM", "VRM", "VRM4Uで読み込む用（MToonマテリアル付き）"),
               ("FBX", "FBX", "普通のスケルタルメッシュとして読み込む用")],
        default="VRM")


def _vrm_addon_available():
    return hasattr(bpy.ops.import_scene, "vrm") and hasattr(bpy.ops.export_scene, "vrm")


class VCT_OT_import_vrm(bpy.types.Operator):
    bl_idname = "vct.import_vrm"
    bl_label = "VRMを読み込む"
    bl_description = "VRMファイルを読み込む（VRM拡張機能の読み込み画面を開く）"

    def execute(self, context):
        if not _vrm_addon_available():
            self.report({"ERROR"}, "VRM拡張機能（VRM Add-on for Blender）を有効にしてください")
            return {"CANCELLED"}
        return bpy.ops.import_scene.vrm("INVOKE_DEFAULT")


class VCT_OT_export_vrm(bpy.types.Operator):
    bl_idname = "vct.export_vrm"
    bl_label = "VRMを書き出す"
    bl_description = "表示中のパーツだけを含めてVRMを書き出す（非表示のパーツは含まれない）"

    def execute(self, context):
        if not _vrm_addon_available():
            self.report({"ERROR"}, "VRM拡張機能（VRM Add-on for Blender）を有効にしてください")
            return {"CANCELLED"}
        arm = core.find_armature(context)
        if arm is None:
            self.report({"ERROR"}, "VRMのアーマチュアが見つかりません")
            return {"CANCELLED"}
        return bpy.ops.export_scene.vrm("INVOKE_DEFAULT", armature_object_name=arm.name)


class VCT_OT_split_parts(bpy.types.Operator):
    bl_idname = "vct.split_parts"
    bl_label = "パーツに分ける"
    bl_description = "マテリアルの種類ごとにメッシュを分ける（体・上着・ズボン・靴など）。表情を持つ顔はそのまま"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = core.find_armature(context)
        if arm is None:
            self.report({"ERROR"}, "VRMのアーマチュアが見つかりません")
            return {"CANCELLED"}
        parts = core.split_parts(context, arm)
        self.report({"INFO"}, "パーツ: " + ", ".join(parts))
        return {"FINISHED"}


class VCT_OT_apply_color(bpy.types.Operator):
    bl_idname = "vct.apply_color"
    bl_label = "テクスチャに適用"
    bl_description = "色相・彩度・明度の調整をテクスチャに焼き込む（元の画像は残る）"
    bl_options = {"REGISTER", "UNDO"}

    material_name: bpy.props.StringProperty()

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is None:
            return {"CANCELLED"}
        n = texture.rebuild(mat)
        if n == 0 and texture.has_edits(mat):
            self.report({"WARNING"}, "このマテリアルには調整できるテクスチャがありません")
        return {"FINISHED"}


class VCT_OT_reset_color(bpy.types.Operator):
    bl_idname = "vct.reset_color"
    bl_label = "元に戻す"
    bl_description = "テクスチャを元の画像に戻し、色相・彩度・明度の値もリセットする"
    bl_options = {"REGISTER", "UNDO"}

    material_name: bpy.props.StringProperty()

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is None:
            return {"CANCELLED"}
        mat.vct.hue, mat.vct.saturation, mat.vct.value = 0.0, 1.0, 1.0
        texture.rebuild(mat)   # 部分の色替えやエンブレムがあればそれは残る
        return {"FINISHED"}


class VCT_OT_import_donor(bpy.types.Operator, ImportHelper):
    bl_idname = "vct.import_donor"
    bl_label = "別のVRMを読み込む"
    bl_description = "パーツを持ってくる元のVRMを、横に並べて読み込む"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".vrm"
    filter_glob: StringProperty(default="*.vrm", options={"HIDDEN"})

    def execute(self, context):
        if not _vrm_addon_available():
            self.report({"ERROR"}, "VRM拡張機能（VRM Add-on for Blender）を有効にしてください")
            return {"CANCELLED"}
        target = core.find_armature(context)
        stem = os.path.splitext(os.path.basename(self.filepath))[0]
        donor = swap.import_donor(context, self.filepath, stem)
        if target is not None:
            context.view_layer.objects.active = target
        self.report({"INFO"}, f"{donor.name} を読み込みました")
        return {"FINISHED"}


class VCT_OT_attach_part(bpy.types.Operator):
    bl_idname = "vct.attach_part"
    bl_label = "付ける"
    bl_description = "このパーツを編集中のキャラに付ける（体型の違いを合わせ、揺れ骨と揺れの設定もコピーする）"
    bl_options = {"REGISTER", "UNDO"}

    part_name: StringProperty()

    def execute(self, context):
        target = core.find_armature(context)
        part = bpy.data.objects.get(self.part_name)
        if target is None or part is None:
            self.report({"ERROR"}, "キャラかパーツが見つかりません")
            return {"CANCELLED"}
        settings = context.scene.vct
        new, bones, springs = swap.attach_part(
            context, part, target, settings.hide_same_kind, settings.push_out)
        context.view_layer.objects.active = target
        self.report({"INFO"}, f"{new.name} を付けました（揺れ骨 {bones} 本、揺れ設定 {springs} 個）")
        return {"FINISHED"}


class VCT_OT_push_out(bpy.types.Operator):
    bl_idname = "vct.push_out"
    bl_label = "はみ出しを直す"
    bl_description = "このパーツ（服）を、体が突き抜けている所で外側へ押し出す"
    bl_options = {"REGISTER", "UNDO"}

    part_name: StringProperty()

    def execute(self, context):
        target = core.find_armature(context)
        part = bpy.data.objects.get(self.part_name)
        body = swap.find_body(target) if target else None
        if part is None or body is None:
            self.report({"ERROR"}, "パーツか体（Body）が見つかりません")
            return {"CANCELLED"}
        n = swap.push_out_of_body(context, part, body)
        self.report({"INFO"}, f"{n} 個の頂点を押し出しました")
        return {"FINISHED"}


class VCT_OT_remove_donor(bpy.types.Operator):
    bl_idname = "vct.remove_donor"
    bl_label = "片付ける"
    bl_description = "読み込んだVRMを削除する（付けたパーツは残る）"
    bl_options = {"REGISTER", "UNDO"}

    donor_name: StringProperty()

    def execute(self, context):
        donor = bpy.data.objects.get(self.donor_name)
        if donor is None:
            return {"CANCELLED"}
        swap.remove_donor(donor)
        return {"FINISHED"}


class VCT_OT_export_parts(bpy.types.Operator):
    bl_idname = "vct.export_parts"
    bl_label = "パーツごとに書き出す"
    bl_description = "表示中のパーツを1つずつ、共通の骨格と一緒に書き出す（UEのパーツ集用）"

    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        target = core.find_armature(context)
        if target is None:
            return {"CANCELLED"}
        written = swap.export_parts(context, target, self.directory, context.scene.vct.export_format)
        self.report({"INFO"}, f"{len(written)} 個のパーツを書き出しました")
        return {"FINISHED"}


class VCT_OT_apply_body(bpy.types.Operator):
    bl_idname = "vct.apply_body"
    bl_label = "体型を適用"
    bl_description = "スライダーの体型を、骨とすべてのパーツのメッシュに反映する"
    bl_options = {"REGISTER", "UNDO"}

    reset: BoolProperty(default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        arm = core.find_armature(context)
        if arm is None:
            return {"CANCELLED"}
        if self.reset:
            for prop, *_ in shape.BODY_PARAMS:
                setattr(arm.data.vct_body, prop, 1.0)
        shape.apply_body_shape(context, arm, shape.body_values(arm))
        return {"FINISHED"}


class VCT_OT_create_face_keys(bpy.types.Operator):
    bl_idname = "vct.create_face_keys"
    bl_label = "顔の形スライダーを作る"
    bl_description = "目・眉・口・輪郭の形を変えるシェイプキーを、顔のメッシュに自動で作る"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = core.find_armature(context)
        face = shape.find_face(arm) if arm else None
        if face is None:
            self.report({"ERROR"}, "顔のメッシュが見つかりません")
            return {"CANCELLED"}
        try:
            n = shape.create_face_keys(face)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{n} 個のスライダーを作りました")
        return {"FINISHED"}


class VCT_OT_bake_face_keys(bpy.types.Operator):
    bl_idname = "vct.bake_face_keys"
    bl_label = "顔の形を確定"
    bl_description = ("今の顔の形を基本形に焼き込み、スライダーを0に戻す。"
                      "VRMの書き出しは基本形を使うので、形を固定して書き出すときに押す")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = core.find_armature(context)
        face = shape.find_face(arm) if arm else None
        if face is None:
            return {"CANCELLED"}
        n = shape.bake_face_keys(face)
        self.report({"INFO"}, f"{n} 個のスライダーの形を確定しました")
        return {"FINISHED"}


def _face_or_report(op, context):
    arm = core.find_armature(context)
    face = shape.find_face(arm) if arm else None
    if face is None or face.data.shape_keys is None:
        op.report({"ERROR"}, "表情用のシェイプキーを持つ顔のメッシュが見つかりません")
        return None, None
    return arm, face


class VCT_OT_expr_load(bpy.types.Operator):
    bl_idname = "vct.expr_load"
    bl_label = "編集"
    bl_description = "この表情をミキサーに読み込んで編集する"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty()

    def execute(self, context):
        arm, face = _face_or_report(self, context)
        if face is None:
            return {"CANCELLED"}
        expr.load_to_mix(arm, face, self.name)
        context.scene.vct.expr_editing = self.name
        return {"FINISHED"}


class VCT_OT_expr_reset(bpy.types.Operator):
    bl_idname = "vct.expr_reset"
    bl_label = "リセット"
    bl_description = "ミキサーの値と表情のプレビューをすべて0に戻す"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm, face = _face_or_report(self, context)
        if face is None:
            return {"CANCELLED"}
        expr.reset_mix(arm, face)
        context.scene.vct.expr_editing = ""
        return {"FINISHED"}


class VCT_OT_expr_save(bpy.types.Operator):
    bl_idname = "vct.expr_save"
    bl_label = "保存"
    bl_description = "今のミキサーの値を表情として保存する"
    bl_options = {"REGISTER", "UNDO"}

    overwrite: BoolProperty(default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        arm, face = _face_or_report(self, context)
        if face is None:
            return {"CANCELLED"}
        settings = context.scene.vct
        if self.overwrite:
            if not settings.expr_editing:
                return {"CANCELLED"}
            name = expr.save_mix(arm, face, settings.expr_editing, create_custom=False)
        else:
            if not settings.expr_name.strip():
                self.report({"ERROR"}, "表情の名前を入れてください")
                return {"CANCELLED"}
            name = expr.save_mix(arm, face, settings.expr_name.strip(), create_custom=True)
        settings.expr_editing = name
        self.report({"INFO"}, f"表情「{name}」を保存しました")
        return {"FINISHED"}


class VCT_OT_expr_remove(bpy.types.Operator):
    bl_idname = "vct.expr_remove"
    bl_label = "削除"
    bl_description = "このカスタム表情を削除する"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty()

    def execute(self, context):
        arm = core.find_armature(context)
        if arm is None:
            return {"CANCELLED"}
        expr.remove_custom(arm, self.name)
        if context.scene.vct.expr_editing == self.name:
            context.scene.vct.expr_editing = ""
        return {"FINISHED"}


# ---- テクスチャ編集 -----------------------------------------------------------
def _tex_target(op, context):
    mat = context.scene.vct.tex_material
    if mat is None:
        op.report({"ERROR"}, "編集するマテリアルを選んでください")
    return mat


def _parts_with_material(context, mat):
    arm = core.find_armature(context)
    return [o for o in core.part_objects(arm) if mat in o.data.materials[:]] if arm else []


class VCT_OT_tex_pick(bpy.types.Operator):
    bl_idname = "vct.tex_pick"
    bl_label = "モデルをクリック"
    bl_description = "3Dビューでモデルをクリックして、マテリアルやエンブレムの位置を選ぶ（右クリック/Escで中止）"
    bl_options = {"REGISTER", "UNDO"}

    target: EnumProperty(items=[("MATERIAL", "マテリアル", ""), ("DECAL", "エンブレム", "")])
    index: IntProperty(default=0)

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "3Dビューから使ってください")
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set("EYEDROPPER")
        context.area.header_text_set("モデルをクリック（右クリック / Esc で中止）")
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        context.window.cursor_modal_restore()
        context.area.header_text_set(None)

    def modal(self, context, event):
        if event.type in {"RIGHTMOUSE", "ESC"}:
            self._finish(context)
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"PASS_THROUGH"}
        region = next((r for r in context.area.regions if r.type == "WINDOW"
                       and r.x <= event.mouse_x < r.x + r.width
                       and r.y <= event.mouse_y < r.y + r.height), None)
        if region is None:
            return {"RUNNING_MODAL"}
        rv3d = context.area.spaces.active.region_3d
        coord = (event.mouse_x - region.x, event.mouse_y - region.y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        arm = core.find_armature(context)
        hit = texture.pick_uv(context, core.part_objects(arm), origin, direction) if arm else None
        if hit is None:
            return {"RUNNING_MODAL"}
        _obj, mat, uv = hit
        settings = context.scene.vct
        if self.target == "MATERIAL":
            settings.tex_material = mat
        else:
            cur = settings.tex_material
            if cur is None or mat != cur or self.index >= len(cur.vct.decals):
                self.report({"WARNING"}, "選んでいるマテリアルの上をクリックしてください")
                return {"RUNNING_MODAL"}
            d = cur.vct.decals[self.index]
            d.u, d.v = uv
            texture.rebuild(cur)
        self._finish(context)
        return {"FINISHED"}


class VCT_OT_region_add(bpy.types.Operator):
    bl_idname = "vct.region_add"
    bl_label = "部分を追加"
    bl_description = "色を変える部分（マスク）を追加する"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        r = mat.vct.regions.add()
        r.name = f"部分{len(mat.vct.regions)}"
        texture.ensure_mask(mat, r)
        return {"FINISHED"}


class VCT_OT_region_remove(bpy.types.Operator):
    bl_idname = "vct.region_remove"
    bl_label = "削除"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        mask = mat.vct.regions[self.index].mask
        mat.vct.regions.remove(self.index)
        if mask is not None and mask.users == 0:
            bpy.data.images.remove(mask)
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_region_from_faces(bpy.types.Operator):
    bl_idname = "vct.region_from_faces"
    bl_label = "選択した面から"
    bl_description = "編集モードで選んだ面（このマテリアルの部分）をマスクに塗る"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        if context.object is not None and context.object.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        mode = context.scene.vct.mask_mode
        if mode == "INTERSECT":
            mode = "ADD"
        n = texture.mask_from_faces(mat, mat.vct.regions[self.index],
                                    _parts_with_material(context, mat), mode)
        if n == 0:
            self.report({"WARNING"}, "このマテリアルの面が選ばれていません（編集モードで面を選んでから押してください）")
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_region_from_color(bpy.types.Operator):
    bl_idname = "vct.region_from_color"
    bl_label = "色で選ぶ"
    bl_description = "元のテクスチャで「選ぶ色」に近い画素をマスクにする"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        n = texture.mask_from_color(mat, mat.vct.regions[self.index], context.scene.vct.mask_mode)
        self.report({"INFO"}, f"{n} 画素が選ばれました")
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_region_paint(bpy.types.Operator):
    bl_idname = "vct.region_paint"
    bl_label = "手で塗る"
    bl_description = "テクスチャペイントでマスクを塗る（白=色を変える、黒=変えない）。終わったら「塗り終わり」を押す"
    bl_options = {"REGISTER"}

    index: IntProperty()

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        parts = [o for o in _parts_with_material(context, mat) if not o.hide_viewport]
        if not parts:
            self.report({"ERROR"}, "このマテリアルを使う表示中のパーツがありません")
            return {"CANCELLED"}
        mask = texture.ensure_mask(mat, mat.vct.regions[self.index])
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj = parts[0]
        core._select_only(context, [obj], obj)
        obj.active_material_index = obj.data.materials[:].index(mat)
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        ip = context.scene.tool_settings.image_paint
        ip.mode = "IMAGE"
        ip.canvas = mask
        for owner in (context.scene.tool_settings, ip):
            ups = getattr(owner, "unified_paint_settings", None)
            if ups is not None:
                ups.use_unified_color = True
                ups.color = (1.0, 1.0, 1.0)
                break
        space = context.space_data
        if space is not None and space.type == "VIEW_3D" and space.shading.type == "SOLID":
            space.shading.color_type = "TEXTURE"
        context.scene.vct.painting = True
        return {"FINISHED"}


class VCT_OT_region_paint_done(bpy.types.Operator):
    bl_idname = "vct.region_paint_done"
    bl_label = "塗り終わり"
    bl_description = "テクスチャペイントを終えて、塗ったマスクを色替えに反映する"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.scene.vct.painting = False
        mat = context.scene.vct.tex_material
        if mat is not None:
            for r in mat.vct.regions:
                if r.mask is not None and r.mask.is_dirty:
                    r.mask.pack()
            texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_decal_add(bpy.types.Operator, ImportHelper):
    bl_idname = "vct.decal_add"
    bl_label = "エンブレムを追加"
    bl_description = "画像（透過PNGなど）を読み込んで、エンブレムとして貼る"
    bl_options = {"REGISTER", "UNDO"}

    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.tga;*.webp", options={"HIDDEN"})

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        img = bpy.data.images.load(self.filepath, check_existing=True)
        img.pack()
        d = mat.vct.decals.add()
        d.name = os.path.splitext(os.path.basename(self.filepath))[0]
        d.image = img
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_decal_remove(bpy.types.Operator):
    bl_idname = "vct.decal_remove"
    bl_label = "削除"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        mat.vct.decals.remove(self.index)
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_tex_rebuild(bpy.types.Operator):
    bl_idname = "vct.tex_rebuild"
    bl_label = "反映"
    bl_description = "テクスチャを作り直す"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        texture.rebuild(mat)
        return {"FINISHED"}


class VCT_OT_masks_save(bpy.types.Operator):
    bl_idname = "vct.masks_save"
    bl_label = "マスクをPNGで書き出す"
    bl_description = "このマテリアルの部分マスクをPNGで書き出す（UEのキャラクリで色を選べる範囲に使う）"

    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        mat = _tex_target(self, context)
        if mat is None:
            return {"CANCELLED"}
        written = texture.save_masks(mat, self.directory)
        self.report({"INFO"}, f"{len(written)} 枚のマスクを書き出しました")
        return {"FINISHED"}


class VCT_PT_main(bpy.types.Panel):
    bl_idname = "VCT_PT_main"
    bl_label = "VRMキャラ編集"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"

    def draw(self, context):
        layout = self.layout
        if not _vrm_addon_available():
            layout.label(text="VRM拡張機能を有効にしてください", icon="ERROR")
        row = layout.row(align=True)
        row.operator(VCT_OT_import_vrm.bl_idname, icon="IMPORT")
        row.operator(VCT_OT_export_vrm.bl_idname, icon="EXPORT")

        arm = core.find_armature(context)
        if arm is None:
            layout.label(text="VRMを読み込んでください")
            return
        layout.label(text=arm.name, icon="ARMATURE_DATA")
        layout.operator(VCT_OT_split_parts.bl_idname, icon="MOD_EXPLODE")


class VCT_PT_parts(bpy.types.Panel):
    bl_idname = "VCT_PT_parts"
    bl_label = "パーツと色"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"
    bl_parent_id = "VCT_PT_main"

    @classmethod
    def poll(cls, context):
        return core.find_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        arm = core.find_armature(context)
        for obj in core.part_objects(arm):
            box = layout.box()
            row = box.row(align=True)
            row.prop(obj.vct, "expanded", text="", emboss=False,
                     icon="TRIA_DOWN" if obj.vct.expanded else "TRIA_RIGHT")
            row.prop(obj, "hide_viewport", text="", emboss=False)
            row.prop(obj, "name", text="")
            if not obj.vct.expanded:
                continue
            if obj.get("vct_part") not in swap.NO_PUSH_KINDS:
                box.operator(VCT_OT_push_out.bl_idname, icon="MOD_SHRINKWRAP").part_name = obj.name
            for mat in obj.data.materials:
                if mat is None:
                    continue
                self.draw_material(box, mat)

    @staticmethod
    def draw_material(layout, mat):
        col = layout.column(align=True)
        col.label(text=mat.name.replace(" (Instance)", ""), icon="MATERIAL")
        mt = core.mtoon(mat)
        if mt is None:
            col.label(text="MToonではないマテリアル")
            return
        col.prop(mt.pbr_metallic_roughness, "base_color_factor", text="色（乗算）")
        col.prop(mt.extensions.vrmc_materials_mtoon, "shade_color_factor", text="影色（乗算）")
        col.separator()
        col.prop(mat.vct, "hue")
        col.prop(mat.vct, "saturation")
        col.prop(mat.vct, "value")
        row = col.row(align=True)
        row.operator(VCT_OT_apply_color.bl_idname, icon="BRUSH_DATA").material_name = mat.name
        row.operator(VCT_OT_reset_color.bl_idname, icon="LOOP_BACK").material_name = mat.name


class VCT_PT_swap(bpy.types.Panel):
    bl_idname = "VCT_PT_swap"
    bl_label = "パーツの差し替え"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"
    bl_parent_id = "VCT_PT_main"

    @classmethod
    def poll(cls, context):
        return core.find_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.vct
        layout.operator(VCT_OT_import_donor.bl_idname, icon="IMPORT")
        layout.prop(settings, "hide_same_kind")
        layout.prop(settings, "push_out")
        for donor in swap.donor_armatures(context.scene):
            box = layout.box()
            row = box.row()
            row.label(text=donor.name, icon="ARMATURE_DATA")
            row.operator(VCT_OT_remove_donor.bl_idname, text="", icon="TRASH").donor_name = donor.name
            for part in core.part_objects(donor):
                row = box.row(align=True)
                row.label(text=part.get("vct_part", part.name))
                row.operator(VCT_OT_attach_part.bl_idname, icon="LINKED").part_name = part.name

        layout.separator()
        row = layout.row(align=True)
        row.prop(settings, "export_format", text="")
        row.operator(VCT_OT_export_parts.bl_idname, icon="EXPORT")


class VCT_PT_shape(bpy.types.Panel):
    bl_idname = "VCT_PT_shape"
    bl_label = "体型・顔の形"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"
    bl_parent_id = "VCT_PT_main"

    @classmethod
    def poll(cls, context):
        return core.find_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        arm = core.find_armature(context)

        box = layout.box()
        box.label(text="体型", icon="ARMATURE_DATA")
        col = box.column(align=True)
        for prop, *_ in shape.BODY_PARAMS:
            col.prop(arm.data.vct_body, prop, slider=True)
        row = box.row(align=True)
        row.operator(VCT_OT_apply_body.bl_idname, icon="CHECKMARK")
        op = row.operator(VCT_OT_apply_body.bl_idname, text="元の体型に戻す", icon="LOOP_BACK")
        op.reset = True

        box = layout.box()
        box.label(text="顔の形", icon="USER")
        face = shape.find_face(arm)
        if face is None:
            box.label(text="顔のメッシュが見つかりません")
            return
        keys = face.data.shape_keys
        made = [(keys.key_blocks.get(n), label) for n, label in shape.FACE_KEYS] if keys else []
        made = [(kb, label) for kb, label in made if kb is not None]
        if not made:
            box.operator(VCT_OT_create_face_keys.bl_idname, icon="SHAPEKEY_DATA")
            return
        col = box.column(align=True)
        for kb, label in made:
            col.prop(kb, "value", text=label, slider=True)
        row = box.row(align=True)
        row.operator(VCT_OT_bake_face_keys.bl_idname, icon="CHECKMARK")
        row.operator(VCT_OT_create_face_keys.bl_idname, text="作り直す", icon="FILE_REFRESH")


class VCT_PT_expr(bpy.types.Panel):
    bl_idname = "VCT_PT_expr"
    bl_label = "表情"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"
    bl_parent_id = "VCT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return core.find_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.vct
        arm = core.find_armature(context)
        face = shape.find_face(arm)
        if face is None or face.data.shape_keys is None:
            layout.label(text="表情用のシェイプキーを持つ顔が見つかりません")
            return

        box = layout.box()
        box.label(text="表情（スライダーでプレビュー）", icon="SHAPEKEY_DATA")
        col = box.column(align=True)
        for name, e, custom in expr.all_expressions(arm):
            row = col.row(align=True)
            row.prop(e, "preview", text=name, slider=True)
            row.operator(VCT_OT_expr_load.bl_idname, text="", icon="GREASEPENCIL").name = name
            if custom:
                row.operator(VCT_OT_expr_remove.bl_idname, text="", icon="X").name = name

        box = layout.box()
        editing = settings.expr_editing
        box.label(text=f"ミキサー：{editing}" if editing else "ミキサー：新しい表情", icon="MODIFIER")
        box.row().prop(settings, "expr_group", expand=True)
        col = box.column(align=True)
        for label, kbs in expr.grouped_mixer_keys(face):
            if label != settings.expr_group:
                continue
            for kb in kbs:
                col.prop(kb, "value", text=kb.name.replace("Fcl_", ""), slider=True)
        box.operator(VCT_OT_expr_reset.bl_idname, icon="LOOP_BACK")
        if editing:
            op = box.operator(VCT_OT_expr_save.bl_idname, text=f"「{editing}」に上書き保存", icon="FILE_TICK")
            op.overwrite = True
        row = box.row(align=True)
        row.prop(settings, "expr_name", text="")
        row.operator(VCT_OT_expr_save.bl_idname, text="新しい表情として保存", icon="ADD")


class VCT_PT_texture(bpy.types.Panel):
    bl_idname = "VCT_PT_texture"
    bl_label = "テクスチャ編集"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "キャラ編集"
    bl_parent_id = "VCT_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return core.find_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.vct
        row = layout.row(align=True)
        row.prop(settings, "tex_material", text="")
        row.operator(VCT_OT_tex_pick.bl_idname, text="", icon="EYEDROPPER").target = "MATERIAL"
        mat = settings.tex_material
        if mat is None:
            layout.label(text="マテリアルを選ぶか、スポイトでモデルをクリック")
            return
        if settings.painting:
            box = layout.box()
            box.label(text="マスクを塗っています（白=変える）", icon="BRUSH_DATA")
            box.operator(VCT_OT_region_paint_done.bl_idname, icon="CHECKMARK")
            return

        row = layout.row(align=True)
        row.prop(settings, "tex_live")
        row.operator(VCT_OT_tex_rebuild.bl_idname, icon="FILE_REFRESH")

        # 部分の色替え
        box = layout.box()
        box.label(text="部分の色替え", icon="MOD_MASK")
        box.prop(settings, "mask_mode", expand=True)
        for i, r in enumerate(mat.vct.regions):
            sub = box.box()
            row = sub.row(align=True)
            row.prop(r, "expanded", text="", emboss=False,
                     icon="TRIA_DOWN" if r.expanded else "TRIA_RIGHT")
            row.prop(r, "enabled", text="")
            row.prop(r, "name", text="")
            row.prop(r, "color", text="")
            row.operator(VCT_OT_region_remove.bl_idname, text="", icon="X").index = i
            if not r.expanded:
                continue
            sub.prop(r, "strength", slider=True)
            row = sub.row(align=True)
            row.operator(VCT_OT_region_from_faces.bl_idname, icon="FACESEL").index = i
            row.operator(VCT_OT_region_paint.bl_idname, icon="BRUSH_DATA").index = i
            row = sub.row(align=True)
            row.prop(r, "pick_color", text="")
            row.prop(r, "tolerance", slider=True)
            row.operator(VCT_OT_region_from_color.bl_idname, icon="EYEDROPPER").index = i
        row = box.row(align=True)
        row.operator(VCT_OT_region_add.bl_idname, icon="ADD")
        row.operator(VCT_OT_masks_save.bl_idname, text="", icon="EXPORT")

        # エンブレム
        box = layout.box()
        box.label(text="柄・エンブレム", icon="IMAGE_DATA")
        for i, d in enumerate(mat.vct.decals):
            sub = box.box()
            row = sub.row(align=True)
            row.prop(d, "expanded", text="", emboss=False,
                     icon="TRIA_DOWN" if d.expanded else "TRIA_RIGHT")
            row.prop(d, "enabled", text="")
            row.prop(d, "name", text="")
            op = row.operator(VCT_OT_tex_pick.bl_idname, text="", icon="RESTRICT_SELECT_OFF")
            op.target, op.index = "DECAL", i
            row.operator(VCT_OT_decal_remove.bl_idname, text="", icon="X").index = i
            if not d.expanded:
                continue
            col = sub.column(align=True)
            col.prop(d, "image", text="")
            row = col.row(align=True)
            row.prop(d, "u")
            row.prop(d, "v")
            col.prop(d, "scale", slider=True)
            col.prop(d, "rotation")
            col.prop(d, "opacity", slider=True)
            row = col.row(align=True)
            row.prop(d, "tint", text="")
            row.prop(d, "flip", toggle=True)
        box.operator(VCT_OT_decal_add.bl_idname, icon="ADD")


classes = (
    *texture.PROPERTY_CLASSES,
    VCT_MaterialSettings,
    VCT_ObjectSettings,
    VCT_BodyShape,
    VCT_SceneSettings,
    VCT_OT_import_vrm,
    VCT_OT_export_vrm,
    VCT_OT_split_parts,
    VCT_OT_apply_color,
    VCT_OT_reset_color,
    VCT_OT_import_donor,
    VCT_OT_attach_part,
    VCT_OT_push_out,
    VCT_OT_remove_donor,
    VCT_OT_export_parts,
    VCT_OT_apply_body,
    VCT_OT_create_face_keys,
    VCT_OT_bake_face_keys,
    VCT_OT_expr_load,
    VCT_OT_expr_reset,
    VCT_OT_expr_save,
    VCT_OT_expr_remove,
    VCT_OT_tex_pick,
    VCT_OT_region_add,
    VCT_OT_region_remove,
    VCT_OT_region_from_faces,
    VCT_OT_region_from_color,
    VCT_OT_region_paint,
    VCT_OT_region_paint_done,
    VCT_OT_decal_add,
    VCT_OT_decal_remove,
    VCT_OT_tex_rebuild,
    VCT_OT_masks_save,
    VCT_PT_main,
    VCT_PT_parts,
    VCT_PT_swap,
    VCT_PT_shape,
    VCT_PT_expr,
    VCT_PT_texture,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.vct = PointerProperty(type=VCT_MaterialSettings)
    bpy.types.Object.vct = PointerProperty(type=VCT_ObjectSettings)
    bpy.types.Scene.vct = PointerProperty(type=VCT_SceneSettings)
    bpy.types.Armature.vct_body = PointerProperty(type=VCT_BodyShape)


def unregister():
    del bpy.types.Armature.vct_body
    del bpy.types.Scene.vct
    del bpy.types.Object.vct
    del bpy.types.Material.vct
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
