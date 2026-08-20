def convert_objects_to_js(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        # objects = [line.strip() for line in f if line.strip()]
        objects = [line.strip().replace(" ", "_") for line in f if line.strip()]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("export const icons = [\n")
        for obj in objects:
            f.write(f'  "{obj}",\n')
        f.write("];\n")


convert_objects_to_js("./dict/tag/tag_list.txt", "./frontend/src/helper/words.js")
