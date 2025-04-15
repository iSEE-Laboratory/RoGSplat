from yacs.config import CfgNode as CN


class Config:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.stage1_ckpt = None
        self.cfg.stage2_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.depth_ckpt=None
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.stage_1_num_steps = 0
        self.cfg.depth_refine_num_steps=0
        self.cfg.stage_2_num_steps = 0
        self.cfg.coarse_vox_size=0.02
        self.cfg.fine_vox_size=0.01

        self.cfg.dataset = CN()
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_processed_data = None
        self.cfg.dataset.data_root = ''
        self.cfg.dataset.training_view_num= 0
        self.cfg.dataset.ref_view = []
        self.cfg.dataset.ref_view_num = 4
        self.cfg.dataset.valid_ref_view_num = 4
        self.cfg.dataset.ratio = 1.0
        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 1000.0
        self.cfg.dataset.znear = 0.1
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0



        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0

        self.cfg.model=CN()

        self.cfg.model.decoder=CN()

        self.cfg.model.decoder.name="splatting_cuda"

        self.cfg.model.encoder=CN()
        self.cfg.model.encoder.name="epipolar"
        self.cfg.model.encoder.gsnet = CN()
        self.cfg.model.encoder.gsnet.encoder_dims = None
        self.cfg.model.encoder.gsnet.decoder_dims = None
        self.cfg.model.encoder.gsnet.parm_head_dim = None
        self.cfg.model.encoder.opacity_mapping=CN()
        self.cfg.model.encoder.opacity_mapping.initial= 0.0
        self.cfg.model.encoder.opacity_mapping.final= 0.0
        self.cfg.model.encoder.opacity_mapping.warm_up= 1


        self.cfg.model.encoder.num_monocular_samples= 32
        self.cfg.model.encoder.num_surfaces= 1
        self.cfg.model.encoder.predict_opacity= False
        self.cfg.model.encoder.near_disparity= 3.0

        self.cfg.model.encoder.gaussians_per_pixel= 3

        self.cfg.model.encoder.gaussian_adapter=CN()
        self.cfg.model.encoder.gaussian_adapter.gaussian_scale_min= 0.5
        self.cfg.model.encoder.gaussian_adapter.gaussian_scale_max= 15.0
        self.cfg.model.encoder.gaussian_adapter.sh_degree= 4

        self.cfg.model.encoder.d_feature= 128

        self.cfg.model.encoder.epipolar_transformer=CN()
        self.cfg.model.encoder.epipolar_transformer.self_attention=CN()
        self.cfg.model.encoder.epipolar_transformer.self_attention.patch_size= 4
        self.cfg.model.encoder.epipolar_transformer.self_attention.num_octaves= 10
        self.cfg.model.encoder.epipolar_transformer.self_attention.num_layers= 2
        self.cfg.model.encoder.epipolar_transformer.self_attention.num_heads= 4
        self.cfg.model.encoder.epipolar_transformer.self_attention.d_token= 128
        self.cfg.model.encoder.epipolar_transformer.self_attention.d_dot= 128
        self.cfg.model.encoder.epipolar_transformer.self_attention.d_mlp= 256

        self.cfg.model.encoder.epipolar_transformer.num_octaves= 10
        self.cfg.model.encoder.epipolar_transformer.num_layers= 2
        self.cfg.model.encoder.epipolar_transformer.num_heads= 4
        self.cfg.model.encoder.epipolar_transformer.num_samples= 32
        self.cfg.model.encoder.epipolar_transformer.d_dot= 128
        self.cfg.model.encoder.epipolar_transformer.d_mlp= 256
        self.cfg.model.encoder.epipolar_transformer.downscale= 4

        self.cfg.model.encoder.visualizer=CN()
        self.cfg.model.encoder.visualizer.num_samples= 8
        self.cfg.model.encoder.visualizer.min_resolution= 256
        self.cfg.model.encoder.visualizer.export_ply= False

        self.cfg.model.encoder.apply_bounds_shim= True

        # Use this to ablate the epipolar transformer.
        self.cfg.model.encoder.use_epipolar_transformer= True

        self.cfg.model.encoder.use_transmittance= False

        self.cfg.model.encoder.backbone=CN()
        self.cfg.model.encoder.backbone.name= "dino"
        self.cfg.model.encoder.backbone.backbone=CN()
        self.cfg.model.encoder.backbone.backbone.model= "dino_vitb8"
        self.cfg.model.encoder.backbone.backbone.d_out= 512

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()
