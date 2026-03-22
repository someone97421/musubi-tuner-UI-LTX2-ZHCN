import codecs

content = codecs.open("ltx2_ui_zh.py", "r", "utf-8").read()

# 1. Update run_cache_latents definition to include memory parameters
old_run_cache = """def run_cache_latents(dataset_cfg, ltx2_ckpt, vae_p, vae_dtype_c, skip_existing, fp8_vae):
    cmd = [sys.executable, "ltx2_cache_latents.py",
           "--dataset_config", dataset_cfg or "dataset_ltx2.toml"]
    if ltx2_ckpt: cmd += ["--ltx2_checkpoint", ltx2_ckpt]
    if vae_p: cmd += ["--vae", vae_p]
    if vae_dtype_c: cmd += ["--vae_dtype", vae_dtype_c]
    if skip_existing: cmd += ["--skip_existing"]
    if fp8_vae: cmd += ["--fp8_vae"]
    start_subprocess(cmd)
    return "已启动 Cache Latents，请查看调试框。"
"""

new_run_cache = """def run_cache_latents(dataset_cfg, ltx2_ckpt, vae_p, vae_dtype_c, skip_existing, fp8_vae, v_chunk, v_sp_tile, v_sp_ov, v_tp_tile, v_tp_ov):
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
content = content.replace(old_run_cache, new_run_cache)

# 2. Update the UI block for cache latents
old_ui_block = """                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        c_vae_dtype = gr.Dropdown(label="VAE Dtype（覆盖全局）",
                            choices=["", "fp16", "bf16", "fp32"], value="bf16")
                        c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        cache_latents_btn = gr.Button("▶ 运行 Cache Latents", variant="primary")"""

new_ui_block = """                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        with gr.Row():
                            c_vae_dtype = gr.Dropdown(label="VAE Dtype", choices=["", "fp16", "bf16", "fp32"], value="bf16")
                            c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                            c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        with gr.Accordion("🛠️ 内存优化 (VAE Chunking/Tiling)", open=False):
                            gr.Markdown("如果显存爆炸 (OOM)，请设置 `Chunk Size` 或 `Spatial Tile` 进行显存分块压缩（牺牲少许速度降显存）。例如：Chunk 设为 16 或 Spatial 设为 512。")
                            with gr.Row():
                                c_vae_chunk = gr.Number(label="Temporal Chunk Size", value=0, precision=0, info="16或32")
                                c_vae_tp_tile = gr.Number(label="Temporal Tile Size", value=0, precision=0, info="时间维度切片")
                                c_vae_tp_ov = gr.Number(label="Temporal Tile Overlap", value=24, precision=0)
                            with gr.Row():
                                c_vae_sp_tile = gr.Number(label="Spatial Tile Size", value=0, precision=0, info="例如512")
                                c_vae_sp_ov = gr.Number(label="Spatial Tile Overlap", value=64, precision=0)
                        cache_latents_btn = gr.Button("▶ 运行 Cache Latents", variant="primary")"""
content = content.replace(old_ui_block, new_ui_block)

# 3. Update the click event for cache latents
old_click = """                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae],
                    outputs=cache_status
                )"""

new_click = """                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae, c_vae_chunk, c_vae_sp_tile, c_vae_sp_ov, c_vae_tp_tile, c_vae_tp_ov],
                    outputs=cache_status
                )"""
content = content.replace(old_click, new_click)

codecs.open("ltx2_ui_zh_mod2.py", "w", "utf-8").write(content)
print("Finished rewriting ui_zh to version 2.")
