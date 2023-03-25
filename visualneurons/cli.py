"""Command line utility for style transfer"""
import click
from prototypes.styletransfer.model import make_blog_style_transfer
from prototypes.styletransfer.images import load_image, save_image

@click.group()
def main():
    pass


@main.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--iterations', default=100)
@click.option('--output_image', default='out.png')
def image(content_image, style_image, iterations, output_image):
    click.echo(f"Styling {content_image} using style {style_image}")
    # Load model
    model = make_blog_style_transfer()

    style_image = load_image(style_image)
    content_image = load_image(content_image)
    for result in model.run_style_transfer(style_img=style_image, content_img=content_image, num_iterations=iterations, style_weight=1e-2, total_variation_weight=1e-2):
        print(result.total_loss)

    save_image(result.image, output_image)

if __name__ == '__main__':
    main()