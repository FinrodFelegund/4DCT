from PIL import Image
import glob
import pydicom
import argparse


def read_data(input_path):
    """Reads the full sequence of images for all breathing phases"""
    
    phases = sorted(list(glob.glob(pathname='**', root_dir=input_path)))
    phases_with_slices = []
    for phase in phases:
        slices = glob.glob(pathname='**', root_dir=f'{input_path}{phase}')
        slices_with_metadata = []
        for slice in slices:
            dicom_data = pydicom.dcmread(fp=f'{input_path}{phase}/{slice}')
  
            z_position = dicom_data.ImagePositionPatient[2]
            slices_with_metadata.append((z_position, f'{input_path}{phase}/{slice}'))

        slices_with_metadata.sort(key=lambda x: x[0], reverse=True)
        slices_with_metadata = [Image.fromarray(pydicom.dcmread(path).pixel_array) for _, path in slices_with_metadata]
        phases_with_slices.append(slices_with_metadata)

    return phases_with_slices

def generate_gif_full_thorax(input_path, output_path, duration=200):
    """Generate a .gif file of the full thorax from a sequence of CT images."""

    phases_with_slices = read_data(input_path)
    for i, phase in enumerate(phases_with_slices):
        for img in phase:
            img = img.save(
                f'{output_path}/phase{i}.gif',
                format='GIF',
                append_images=phase[1:],
                save_all=True,
                duration=duration,
                loop=0
            )
            break



    

def generate_gif_breathing_motion(input_path, output_path, duration=200):
    """Generate a .gif file for a breathing phase from a sequence of CT Images."""
    
    phases_with_slices = read_data(input_path)
    phases_with_slices = [list(i) for i in zip(*phases_with_slices)]
    for i, slices in enumerate(phases_with_slices):
        for img in slices:
            img = img.save(
                f'{output_path}/slice{i}.gif',
                format='GIF',
                append_images=slices[1:],
                save_all=True,
                duration=duration,
                loop=0
            )
            break
    
    


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a .gif file from a sequence of CT images.')
    parser.add_argument('--input_path', type=str, help='Path to the input folder of a 4D CT Scan')
    parser.add_argument('--output_path', type=str, help='Path to the output .gif file')
    parser.add_argument('--duration', type=int, default=200, help='Duration of each image in the gif in milliseconds')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generate_gif_breathing_motion(args.input_path, args.output_path, args.duration)
    generate_gif_full_thorax(args.input_path, args.output_path, args.duration)
