# -*- coding: future_fstrings -*-

from argparse import Namespace


cfg = Namespace(batch_size=20, 
                lr=1e-04, 
                num_cell_h=4, 
                num_cell_w=4, 
                max_num_obj=10,
                explained_ratio_threshold=0.2, 
                ratio_anc=1.0, 
                sigma=0.1, 
                size_anc=0.25, 
                var_anc=0.2, 
                var_s=0.15,  
                z_pres_anneal_end_value=0.0001,
                ckpt_dir='./model/', 
                color_num=500,
                cp=1.0, 
                epochs=4000, 
                gen_disc_pres_probs=0.1, 
                generate_freq=5000, 
                last_ckpt='', 
                nocuda=False, 
                num_img_summary=3, 
                observe_frames=5, 
                phase_conv_lstm=True, 
                phase_do_remove_detach=True, 
                phase_eval=True,  
                phase_generate=False, 
                phase_nll=False, 
                phase_no_background=False, 
                phase_parallel=False, 
                phase_simplify_summary=True, 
                print_freq=100, 
                remove_detach_step=30000, 
                save_epoch_freq=1000, 
                seed=666, 
                global_state=0, 
                summary_dir='./summary', 
                tau_end=0.3, 
                tau_ep=20000.0, 
                tau_imp=0.25)

z_what_dim = 4
z_where_scale_dim = 2  # sx sy
z_where_shift_dim = 2  # tx ty
z_pres_dim = 1
glimpse_size = 32
img_h = 64
img_w = img_h
img_encode_dim = 64
z_depth_dim = 1
bg_what_dim = 1

temporal_rnn_hid_dim = 128
temporal_rnn_out_dim = temporal_rnn_hid_dim
propagate_encode_dim = 32
z_where_transit_bias_net_hid_dim = 128
z_depth_transit_net_hid_dim = 128

z_pres_hid_dim = 64
z_what_from_temporal_hid_dim = 64
z_what_enc_dim = 128

prior_rnn_hid_dim = 64
prior_rnn_out_dim = prior_rnn_hid_dim

seq_len = 10
phase_obj_num_contrain = True
phase_rejection = True

temporal_img_enc_hid_dim = 64
temporal_img_enc_dim = 128
z_where_bias_dim = 4
temporal_rnn_inp_dim = 128
prior_rnn_inp_dim = 128
bg_prior_rnn_hid_dim = 32
where_update_scale = .2

pres_logit_factor = 8.8

conv_lstm_hid_dim = 64