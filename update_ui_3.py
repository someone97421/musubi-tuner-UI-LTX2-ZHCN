import codecs

content = codecs.open("ltx2_ui_zh.py", "r", "utf-8").read()

# 1. Update run_cache_latents to check inputs and accept ltx_mode
old_run_cache = """def run_cache_latents(dataset_cfg, ltx2_ckpt, vae_p, vae_dtype_c, skip_existing, fp8_vae, v_chunk, v_sp_tile, v_sp_ov, v_tp_tile, v_tp_ov):
    cmd = [sys.executable, "ltx2_cache_latents.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt: cmd += ["--ltx2_checkpoint", ltx2_ckpt]
    if vae_p: cmd += ["--vae", vae_p]
    if vae_dtype_c: cmd += ["--vae_dtype", vae_dtype_c]
    if skip_existing: cmd += ["--skip_existing"]
    if fp8_vae: cmd += ["--fp8_vae"]
    if v_chunk and v_chunk > 0: cmd += ["--vae_chunk_size", str(int(v_chunk))]
    if v_sp_tile and v_sp_tile > 0: cmd += ["--vae_spatial_tile_size", str(int(v_sp_tile))]
    if v_sp_ov and v_sp_ov > 0: cmd += ["--vae_spatial_tile_overlap", str(int(v_sp_ov))]
    if v_tp_tile and v_tp_tile > 0: cmd += ["--vae_temporal_tile_size", str(int(v_tp_tile))]
    if v_tp_ov and v_tp_ov > 0: cmd += ["--vae_temporal_tile_overlap", str(int(v_tp_ov))]
    start_subprocess(cmd)
    return "已启动 Cache Latents，请查看调试框。"
"""

new_run_cache = """def run_cache_latents(dataset_cfg, ltx2_ckpt, vae_p, vae_dtype_c, skip_existing, fp8_vae, v_chunk, v_sp_tile, v_sp_ov, v_tp_tile, v_tp_ov, c_mode):
    if not ltx2_ckpt and not vae_p:
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【LTX2 权重路径】！"
    
    cmd = [sys.executable, "ltx2_cache_latents.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt: cmd += ["--ltx2_checkpoint", ltx2_ckpt]
    if vae_p: cmd += ["--vae", vae_p]
    if vae_dtype_c: cmd += ["--vae_dtype", vae_dtype_c]
    if skip_existing: cmd += ["--skip_existing"]
    if fp8_vae: cmd += ["--fp8_vae"]
    if v_chunk and v_chunk > 0: cmd += ["--vae_chunk_size", str(int(v_chunk))]
    if v_sp_tile and v_sp_tile > 0: cmd += ["--vae_spatial_tile_size", str(int(v_sp_tile))]
    if v_sp_ov and v_sp_ov > 0: cmd += ["--vae_spatial_tile_overlap", str(int(v_sp_ov))]
    if v_tp_tile and v_tp_tile > 0: cmd += ["--vae_temporal_tile_size", str(int(v_tp_tile))]
    if v_tp_ov and v_tp_ov > 0: cmd += ["--vae_temporal_tile_overlap", str(int(v_tp_ov))]
    if c_mode != "video":
        cmd += ["--ltx2_mode", c_mode]
    start_subprocess(cmd)
    return "已启动 Cache Latents，请查看调试框。"
"""
content = content.replace(old_run_cache, new_run_cache)

# 2. Update run_cache_te
old_run_te = """def run_cache_te(dataset_cfg, ltx2_ckpt, gemma_rt, gemma_sft, te_dtype, skip_existing, load_8bit, load_4bit):
    cmd = [sys.executable, "ltx2_cache_text_encoder_outputs.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt: cmd += ["--ltx2_checkpoint", ltx2_ckpt]
    if gemma_rt: cmd += ["--gemma_root", gemma_rt]
    if gemma_sft: cmd += ["--gemma_safetensors", gemma_sft]
    if te_dtype: cmd += ["--text_encoder_dtype", te_dtype]
    if skip_existing: cmd += ["--skip_existing"]
    if load_8bit: cmd += ["--gemma_load_in_8bit"]
    if load_4bit: cmd += ["--gemma_load_in_4bit"]
    start_subprocess(cmd)
    return "已启动 Cache Text Encoder，请查看调试框。"
"""

new_run_te = """def run_cache_te(dataset_cfg, ltx2_ckpt, gemma_rt, gemma_sft, te_dtype, skip_existing, load_8bit, load_4bit, c_mode):
    if not ltx2_ckpt:
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【LTX2 权重路径】！"
    if not gemma_rt and not gemma_sft:
        return "❌ 错误：请先在界面顶部【0. 全局设置】中填写【Gemma 权重路径】！"
        
    cmd = [sys.executable, "ltx2_cache_text_encoder_outputs.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt: cmd += ["--ltx2_checkpoint", ltx2_ckpt]
    if gemma_rt: cmd += ["--gemma_root", gemma_rt]
    if gemma_sft: cmd += ["--gemma_safetensors", gemma_sft]
    if te_dtype: cmd += ["--mixed_precision", te_dtype]
    if skip_existing: cmd += ["--skip_existing"]
    if load_8bit: cmd += ["--gemma_load_in_8bit"]
    if load_4bit: cmd += ["--gemma_load_in_4bit"]
    if c_mode != "video":
        cmd += ["--ltx2_mode", c_mode]
    start_subprocess(cmd)
    return "已启动 Cache Text Encoder，请查看调试框。"
"""
content = content.replace(old_run_te, new_run_te)

# 3. Update the UI block to pass c_mode
old_ui_block = """            with gr.Accordion("🗂️ 1.2 预缓存视频 Latent 与文本特征", open=True):
                gr.Markdown(\"\"\"
                预缓存视频 Latent 和文本编码器输出，大幅加速训练。  
            **模型路径和数据集配置文件均读取自顶部全局设置，无需重复填写。**
            \"\"\")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        with gr.Row():
                            c_vae_dtype = gr.Dropdown(label="VAE Dtype", choices=["", "fp16", "bf16", "fp32"], value="bf16")
                            c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                            c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        with gr.Accordion("🛠️ 内存优化 (VAE Chunking/Tiling)", open=False):"""

new_ui_block = """            with gr.Accordion("🗂️ 1.2 预缓存视频 Latent 与文本特征", open=True):
                gr.Markdown(\"\"\"
                预缓存视频 Latent 和文本编码器输出，大幅加速训练。  
            **模型路径和数据集配置文件均读取自顶部全局设置，无需重复填写。**
            \"\"\")
                c_mode = gr.Dropdown(label="缓存模式 (--ltx2_mode)", choices=["video", "av", "audio"], value="video", info="要缓存音频特征，请选 av 或 audio。")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        with gr.Row():
                            c_vae_dtype = gr.Dropdown(label="VAE Dtype", choices=["", "fp16", "bf16", "fp32"], value="bf16")
                            c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                            c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        with gr.Accordion("🛠️ 内存优化 (VAE Chunking/Tiling)", open=False):"""
content = content.replace(old_ui_block, new_ui_block)

old_clicks = """                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae, c_vae_chunk, c_vae_sp_tile, c_vae_sp_ov, c_vae_tp_tile, c_vae_tp_ov],
                    outputs=cache_status
                )
                cache_te_btn.click(
                    fn=run_cache_te,
                    inputs=[g_dataset, g_ltx2, g_gemma_root, g_gemma_sft,
                            te_dtype, te_skip, te_8bit, te_4bit],
                    outputs=cache_status
                )"""

new_clicks = """                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae, c_vae_chunk, c_vae_sp_tile, c_vae_sp_ov, c_vae_tp_tile, c_vae_tp_ov, c_mode],
                    outputs=cache_status
                )
                cache_te_btn.click(
                    fn=run_cache_te,
                    inputs=[g_dataset, g_ltx2, g_gemma_root, g_gemma_sft,
                            te_dtype, te_skip, te_8bit, te_4bit, c_mode],
                    outputs=cache_status
                )"""
content = content.replace(old_clicks, new_clicks)

codecs.open("ltx2_ui_zh_mod3.py", "w", "utf-8").write(content)
print("Finished rewriting ui_zh to version 3.")
