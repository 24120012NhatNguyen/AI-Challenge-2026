import yaml
import os


def extract_names():
    """
    Reads object names from a YAML file and writes them to a text file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    yaml_file_path = os.path.join(script_dir, "open-images-v7.yaml")
    output_file_path = os.path.join(script_dir, "obj_name.txt")

    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "names" in data and isinstance(data["names"], dict):
            object_names = data["names"].values()

            with open(output_file_path, "w", encoding="utf-8") as f:
                for name in object_names:
                    f.write(f"{name}\n")

            print(
                f"Successfully extracted {len(object_names)} names to {output_file_path}"
            )
        else:
            print(
                "Error: 'names' key not found or is not in the correct format in the YAML file."
            )

    except FileNotFoundError:
        print(f"Error: The file {yaml_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    extract_names()
