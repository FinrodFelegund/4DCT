from pathlib import Path
import argparse
import dicom2nifti
import os


def convert_4D_CT(patient_dir: str, output_dir: str, reorient: bool = True):
    INPUT_DIR = Path(patient_dir)
    OUTPUT_DIR = Path(output_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for patient_dir in INPUT_DIR.iterdir():
        if not patient_dir.is_dir():
            continue

        for scan_dir in patient_dir.iterdir():
            if not scan_dir.is_dir():
                continue
            
            out_dir_scan = OUTPUT_DIR / patient_dir.name / scan_dir.name
            out_dir_scan.mkdir(parents=True, exist_ok=True)

            index = 0
            for phase_dir in sorted(scan_dir.iterdir()):
                if not phase_dir.is_dir():
                    continue
                output_filename = out_dir_scan / f'phase_0{index}.nii.gz'
                if output_filename.is_file():
                    print(f'File: {output_filename} already exists. Continuing...')
                    continue

                print(f'Converting: {patient_dir.name} | {scan_dir.name} | {phase_dir.name}...')
                try:
                    dicom2nifti.dicom_series_to_nifti(
                        original_dicom_directory=str(phase_dir),
                        output_file=str(output_filename),
                        reorient_nifti=True,

                    )
                    index += 1

                except Exception as e:
                    print(f'Error: failed to convert {phase_dir.name} | {e}')

def rename_nii(input_dir: str):
    input_dir = Path(input_dir)

    for patient_dir in input_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        for scan_dir in patient_dir.iterdir():
            if not scan_dir.is_dir():
                continue

            for i, phase_file in enumerate(sorted(scan_dir.iterdir())):
                if not phase_file.name.endswith('.nii.gz'):
                    continue

                old_name = str(input_dir / patient_dir / scan_dir / phase_file)
                new_name = str(input_dir / patient_dir / scan_dir / f'phase_0{i}.nii.gz')
                os.rename(src=old_name, dst=new_name)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dicom to Nifti conversion")
    parser.add_argument('--input_path', type=str, help='Path to the patient directory')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')

    args = parser.parse_args()

    convert_4D_CT(args.input_path, args.output_path, True)
    rename_nii(args.output_path)