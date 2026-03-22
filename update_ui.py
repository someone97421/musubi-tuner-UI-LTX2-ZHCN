import codecs

content = codecs.open("ltx2_ui_zh.py", "r", "utf-8").read()

# 1. Remove forced light theme from top
content = content.replace('os.environ["GRADIO_THEME"] = "light"', '')

# 2. Modify generate_dataset_toml to include audio parameters
old_gen_toml = """def generate_dataset_toml(res_w, res_h, cap_ext, b_size, e_bucket, no_upscale, vid_dir, cache_dir, t_frames, m_frames, t_fps, save_path):
    try:
        t_frames_list = [int(x.strip()) for x in t_frames.split(",") if x.strip().isdigit()]
        config = {
            "general": {
                "resolution": [int(res_w), int(res_h)],
                "caption_extension": cap_ext,
                "batch_size": int(b_size),
                "enable_bucket": bool(e_bucket),
                "bucket_no_upscale": bool(no_upscale)
            },
            "datasets": [
                {
                    "video_directory": vid_dir,
                    "cache_directory": cache_dir,
                    "target_frames": t_frames_list,
                    "max_frames": int(m_frames),
                    "target_fps": float(t_fps)
                }
            ]
        }
        import toml
        with open(save_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return f"✅ 数据集配置已成功保存至 {save_path}"
    except Exception as e:
        return f"❌ 生成失败: {str(e)}"
"""

new_gen_toml = """def generate_dataset_toml(res_w, res_h, cap_ext, b_size, e_bucket, no_upscale, sep_audio, vid_dir, cache_dir, aud_dir, aud_strat, aud_int, t_frames, m_frames, t_fps, save_path):
    try:
        t_frames_list = [int(x.strip()) for x in t_frames.split(",") if x.strip().isdigit()]
        gen_cfg = {
            "resolution": [int(res_w), int(res_h)],
            "caption_extension": cap_ext,
            "batch_size": int(b_size),
            "enable_bucket": bool(e_bucket),
            "bucket_no_upscale": bool(no_upscale)
        }
        if sep_audio:
            gen_cfg["separate_audio_buckets"] = True

        ds_cfg = {
            "video_directory": vid_dir,
            "cache_directory": cache_dir,
            "target_frames": t_frames_list,
            "max_frames": int(m_frames),
            "target_fps": float(t_fps)
        }
        if aud_dir and aud_dir.strip():
            ds_cfg["audio_directory"] = aud_dir.strip()
            ds_cfg["audio_bucket_strategy"] = aud_strat
            ds_cfg["audio_bucket_interval"] = float(aud_int)

        config = {
            "general": gen_cfg,
            "datasets": [ds_cfg]
        }
        import toml
        with open(save_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        return f"✅ 数据集配置已成功保存至 {save_path}"
    except Exception as e:
        return f"❌ 生成失败: {str(e)}"
"""
content = content.replace(old_gen_toml, new_gen_toml)

# 3. Replace forced light js and blocks
old_blocks_def = """ui_theme = gr.themes.Soft(
    primary_hue="green",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

force_light_js = \"\"\"
function() {
    document.documentElement.classList.remove('dark');
    document.documentElement.classList.add('light');
    if (document.body) {
        document.body.classList.remove('dark');
        document.body.classList.add('light');
    }
}
\"\"\"

with gr.Blocks(title="LTX2 训练控制台", theme=ui_theme, js=force_light_js) as app:"""

new_blocks_def = """ui_theme = gr.themes.Soft(
    primary_hue="green",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="LTX2 训练控制台", theme=ui_theme) as app:"""
content = content.replace(old_blocks_def, new_blocks_def)

# 4. Modify UI Layout for Cache and Audio inputs
old_ui_block = """            with gr.Accordion("📝 1.1 生成数据集配置文件 (TOML)", open=True):
                gr.Markdown("参考 test.toml 的选项，快速生成数据集配置。")
                with gr.Row():
                    ds_res_w = gr.Number(label="Resolution Width", value=960, precision=0)
                    ds_res_h = gr.Number(label="Resolution Height", value=544, precision=0)
                    ds_cap_ext = gr.Textbox(label="Caption Extension", value=".txt")
                with gr.Row():
                    ds_batch_size = gr.Number(label="Batch Size", value=1, precision=0)
                    ds_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
                    ds_bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale", value=False)
                with gr.Row():
                    ds_vid_dir = gr.Textbox(label="Video Directory", value=" ")
                    ds_cache_dir = gr.Textbox(label="Cache Directory", value=" ")
                with gr.Row():
                    ds_target_frames = gr.Textbox(label="Target Frames (逗号分隔)", value="49, 97, 145, 193, 201")
                    ds_max_frames = gr.Number(label="Max Frames", value=201, precision=0)
                    ds_target_fps = gr.Number(label="Target FPS", value=25.0)
                with gr.Row():
                    ds_save_path = gr.Textbox(label="保存路径", value="dataset_ltx2.toml")
                    ds_gen_btn = gr.Button("📄 生成数据集配置", variant="primary")
                ds_gen_status = gr.Textbox(label="状态", interactive=False)
                ds_gen_btn.click(
                    fn=generate_dataset_toml,
                    inputs=[ds_res_w, ds_res_h, ds_cap_ext, ds_batch_size, ds_enable_bucket, ds_bucket_no_upscale,
                            ds_vid_dir, ds_cache_dir, ds_target_frames, ds_max_frames, ds_target_fps, ds_save_path],
                    outputs=ds_gen_status
                )"""

new_ui_block = """            with gr.Accordion("📝 1.1 生成数据集配置文件 (TOML)", open=True):
                gr.Markdown("参考 test.toml 的选项，快速生成数据集配置。")
                with gr.Row():
                    ds_res_w = gr.Number(label="Resolution Width", value=960, precision=0)
                    ds_res_h = gr.Number(label="Resolution Height", value=544, precision=0)
                    ds_cap_ext = gr.Textbox(label="Caption Extension", value=".txt")
                with gr.Row():
                    ds_batch_size = gr.Number(label="Batch Size", value=1, precision=0)
                    ds_enable_bucket = gr.Checkbox(label="Enable Bucket", value=True)
                    ds_bucket_no_upscale = gr.Checkbox(label="Bucket No Upscale", value=False)
                    ds_sep_audio = gr.Checkbox(label="Separate Audio Buckets", value=False)
                with gr.Row():
                    ds_vid_dir = gr.Textbox(label="Video Directory", value=" ")
                    ds_cache_dir = gr.Textbox(label="Cache Directory", value=" ")
                with gr.Row():
                    ds_aud_dir = gr.Textbox(label="Audio Dir (可选, AV模式)", value="")
                    ds_aud_strat = gr.Dropdown(label="Audio Bucket Strategy", choices=["pad", "truncate"], value="pad")
                    ds_aud_int = gr.Number(label="Audio Bucket Interval", value=2.0)
                with gr.Row():
                    ds_target_frames = gr.Textbox(label="Target Frames (逗号分隔)", value="49, 97, 145, 193, 201")
                    ds_max_frames = gr.Number(label="Max Frames", value=201, precision=0)
                    ds_target_fps = gr.Number(label="Target FPS", value=25.0)
                with gr.Row():
                    ds_save_path = gr.Textbox(label="保存路径", value="dataset_ltx2.toml")
                    ds_gen_btn = gr.Button("📄 生成数据集配置", variant="primary")
                ds_gen_status = gr.Textbox(label="状态", interactive=False)
                ds_gen_btn.click(
                    fn=generate_dataset_toml,
                    inputs=[ds_res_w, ds_res_h, ds_cap_ext, ds_batch_size, ds_enable_bucket, ds_bucket_no_upscale,
                            ds_sep_audio, ds_vid_dir, ds_cache_dir, ds_aud_dir, ds_aud_strat, ds_aud_int, 
                            ds_target_frames, ds_max_frames, ds_target_fps, ds_save_path],
                    outputs=ds_gen_status
                )
                
            with gr.Accordion("🗂️ 1.2 预缓存视频 Latent 与文本特征", open=True):
                gr.Markdown(\"\"\"
                预缓存视频 Latent 和文本编码器输出，大幅加速训练。  
            **模型路径和数据集配置文件均读取自顶部全局设置，无需重复填写。**
            \"\"\")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Cache Latents（编码视频特征）")
                        c_vae_dtype = gr.Dropdown(label="VAE Dtype（覆盖全局）",
                            choices=["", "fp16", "bf16", "fp32"], value="bf16")
                        c_skip_lc = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        c_fp8_vae = gr.Checkbox(label="VAE 使用 FP8", value=False)
                        cache_latents_btn = gr.Button("▶ 运行 Cache Latents", variant="primary")

                    with gr.Column():
                        gr.Markdown("#### Cache Text Encoder（编码文本特征）")
                        te_dtype = gr.Dropdown(label="Text Encoder Dtype",
                            choices=["", "fp16", "bf16", "fp32"], value="bf16")
                        te_skip = gr.Checkbox(label="跳过已缓存文件 (--skip_existing)", value=True)
                        with gr.Row():
                            te_8bit = gr.Checkbox(label="Gemma 8bit", value=False)
                            te_4bit = gr.Checkbox(label="Gemma 4bit", value=False)
                        cache_te_btn = gr.Button("▶ 运行 Cache Text Encoder", variant="primary")

                cache_status = gr.Textbox(label="操作状态", interactive=False)

                cache_latents_btn.click(
                    fn=run_cache_latents,
                    inputs=[g_dataset, g_ltx2, g_vae, c_vae_dtype, c_skip_lc, c_fp8_vae],
                    outputs=cache_status
                )
                cache_te_btn.click(
                    fn=run_cache_te,
                    inputs=[g_dataset, g_ltx2, g_gemma_root, g_gemma_sft,
                            te_dtype, te_skip, te_8bit, te_4bit],
                    outputs=cache_status
                )"""

content = content.replace(old_ui_block, new_ui_block)

codecs.open("ltx2_ui_zh_mod.py", "w", "utf-8").write(content)
print("Finished rewriting ui_zh.")
