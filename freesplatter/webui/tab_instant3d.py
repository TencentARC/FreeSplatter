import gradio as gr
from functools import partial
from .gradio_custommodel3d import CustomModel3D
from .shared_opts import create_generate_bar, set_seed
from .gradio_customgs import CustomGS


def create_interface_instant3d(
        grm_api, examples=None):
    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                var_dict['prompt'] = gr.Textbox(
                    label='Prompt', show_label=False, lines=1, placeholder='Prompt', container=False, interactive=True)
                if examples is not None:
                    gr.Examples(
                        examples=examples,
                        inputs=var_dict['prompt'],
                        label='Examples (click one of the items below to start)',
                        api_name=False)
                var_dict['fuse_mesh'] = gr.Checkbox(
                    label='Fuse mesh to get 3D model', value=False, container=False)
                create_generate_bar(var_dict, text='Generate', seed=-1)

            with gr.Column():
                var_dict['out_gs_vis'] = CustomGS(
                    label='Output GS', interactive=False, height=400)
                var_dict['out_gs'] = gr.File(
                    label='Output GS (download)', interactive=False)
                var_dict['out_video'] = gr.Video(
                    label='Output video', interactive=False, autoplay=True, height=320)
                var_dict['out_mesh'] = CustomModel3D(
                    label='Output mesh', interactive=False, height=400)

        var_dict['run_btn'].click(
            fn=set_seed,
            inputs=var_dict['seed'],
            outputs=var_dict['last_seed'], api_name=False
        ).success(
            fn=partial(grm_api, cache_dir=interface.GRADIO_CACHE),
            inputs=[var_dict['last_seed'], var_dict['prompt'], var_dict['fuse_mesh']],
            outputs=[var_dict[k] for k in ['out_gs_vis', 'out_gs', 'out_video', 'out_mesh']], concurrency_id='default_group',
            api_name='run_instant3d'
        )

    return interface, var_dict
