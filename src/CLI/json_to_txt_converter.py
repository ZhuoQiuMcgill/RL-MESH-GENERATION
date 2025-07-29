#!/usr/bin/env python3
"""
JSON to TXT Converter Script

This script converts JSON files containing 2D coordinate arrays to TXT format.
The input JSON file should contain an array of [x, y] coordinate pairs in counter-clockwise order.
The output TXT file will have each coordinate pair on a separate line as "x y" in clockwise order.

IMPORTANT: This script automatically reverses the point order from counter-clockwise (JSON input)
to clockwise (TXT output) as required by the mesh generation system.

Usage:
    python json_to_txt_converter.py -i <input_file_path>

The output file will be created in the same directory as the input file
with the same name but .txt extension.
"""

import json
import argparse
import os
import sys


def convert_json_to_txt(input_file_path):
    """
    Convert a JSON file containing coordinate arrays to TXT format.
    
    The input JSON should contain coordinates in counter-clockwise order.
    The output TXT will have coordinates in clockwise order (reversed).
    
    Args:
        input_file_path (str): Path to the input JSON file
        
    Returns:
        str: Path to the generated TXT file
    """
    try:
        # Read the JSON file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            coordinates = json.load(f)
        
        # Validate that it's a list of coordinate pairs
        if not isinstance(coordinates, list):
            raise ValueError("JSON file must contain an array of coordinates")
        
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, list) or len(coord) != 2:
                raise ValueError(f"Invalid coordinate at index {i}: must be [x, y] pair")
            if not all(isinstance(c, (int, float)) for c in coord):
                raise ValueError(f"Invalid coordinate at index {i}: values must be numbers")
        
        # Generate output file path
        input_dir = os.path.dirname(input_file_path)
        input_name = os.path.basename(input_file_path)
        output_name = os.path.splitext(input_name)[0] + '.txt'
        output_file_path = os.path.join(input_dir, output_name)
        
        # Reverse the coordinate order to convert from counter-clockwise to clockwise
        coordinates_reversed = coordinates[::-1]
        
        print(f"Reversing point order: {len(coordinates)} points (counter-clockwise → clockwise)")
        
        # Write to TXT file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for coord in coordinates_reversed:
                f.write(f"{coord[0]:.3f} {coord[1]:.3f}\n")
        
        return output_file_path
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{input_file_path}': {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to process file '{input_file_path}': {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description='Convert JSON coordinate files to TXT format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python json_to_txt_converter.py -i data/mesh/dolphine3.json
    python json_to_txt_converter.py -i /path/to/coordinates.json

Note:
    The script automatically reverses the point order from counter-clockwise 
    (typical JSON input) to clockwise (required for mesh generation).
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the input JSON file containing coordinate arrays'
    )
    
    args = parser.parse_args()
    
    # Convert the file
    output_path = convert_json_to_txt(args.input)
    print(f"Successfully converted '{args.input}' to '{output_path}'")


if __name__ == '__main__':
    main()