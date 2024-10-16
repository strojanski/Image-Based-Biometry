from PIL import Image
import os
import subprocess as sb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def convert_to_grayscale(input_path):
    output_path = input_path.replace(".bmp", ".png").replace("bmp", "png")
    with Image.open(input_path) as img:
        # Convert the image to 8-bit grayscale ('L' mode in Pillow)
        grayscale_img = img.convert("L")
        grayscale_img.save(output_path, "PNG")
    return output_path


def get_features(f):
    output_file = f.replace(".png", "")
    output_file = f.replace("png", "feats")
    command = ["mindtct", f, output_file]

    if not f.endswith(".png"):
        return

    try:
        result = sb.run(command, check=True)
    except sb.CalledProcessError as e:
        print("Error:", e)


def run_bozorth3(inp1, inp2):
    """Run bozorth3 and capture output"""
    if not inp1.endswith(".xyt") or not inp2.endswith(".xyt"):
        return
    try:
        # Run the bozorth3 command and capture the output
        result = sb.run(
            ["bozorth3", inp1, inp2], stdout=sb.PIPE, stderr=sb.PIPE, check=True
        )

        # Decode the standard output (stdout)
        output = result.stdout.decode("utf-8")
        inp1 = inp1.replace(".xyt", "").replace(".feats", "")
        inp2 = inp2.replace(".xyt", "").replace(".feats", "")
        with open(
            f"data/matches/{inp1.split('/')[-1]}_vs_{inp2.split('/')[-1]}.txt", "w"
        ) as f:
            f.write(output)

        return output
    except sb.CalledProcessError as e:
        # Capture and print the error message if command fails
        error_output = e.stderr.decode("utf-8")
        print("Bozorth3 Error:\n", error_output)
        return None


def get_subjects(path):
    subs = []
    for f in os.listdir(path):
        f = f.split(".")[0]
        subs.append(f)

    return subs


def get_scores(path):
    files = os.listdir(path)

    impostors, genuines = [], []
    imp_names, gen_names = [], []
    all_original_names = []
    all_original_scores = []

    for f in files:
        if not f.endswith(".txt"):
            continue
        sub1 = f.split("_")[0]
        sub2 = f.split("_")[-2]

        fp = open(f"{path}/{f}")
        score = int(fp.read())
        fp.close()

        if sub1 == sub2:
            genuines.append(score)
            gen_names.append((sub1, sub2))
        else:
            impostors.append(score)
            imp_names.append((sub1, sub2))
        all_original_names.append(f)
        all_original_scores.append(score)
    return (
        (impostors, imp_names),
        (genuines, gen_names),
        (all_original_scores, all_original_names),
    )


def plot_bozorth3_comparisons(imp_scores, gen_scores, imp_names, gen_names):
    """
    Plot bozorth3 matching points for impostors and genuines on the same plot.

    Args:
    imp_scores (list): List of matching points for impostors.
    gen_scores (list): List of matching points for genuines.
    imp_names (list of tuples): List of tuples containing impostor subject pairs.
    gen_names (list of tuples): List of tuples containing genuine subject pairs.
    """

    # Convert tuples to string representations for x-axis labels
    imp_labels = ["_".join(name) for name in imp_names]
    gen_labels = ["_".join(name) for name in gen_names]

    # Plot impostor scores in red
    plt.plot([i for i in range(len(imp_scores))], imp_scores, "ro-", label="Impostors")

    # Plot genuine scores in blue
    plt.plot(
        [i for i in range(len(imp_labels), len(imp_labels) + len(gen_labels))],
        gen_scores,
        "bo-",
        label="Genuines",
    )

    # Add labels, title, and formatting
    plt.xlabel("Subject Pairs (sorted alphabetically)", fontsize=25)
    plt.ylabel("Number of Matching Points", fontsize=25)
    plt.title("Bozorth3 Matching Points: Impostors vs Genuines", fontsize=40)

    # Combine labels for x-axis
    all_labels = imp_labels + gen_labels

    # Add a grid and legend
    plt.grid(True)
    plt.legend(fontsize=25)

    # Show the plot
    plt.tight_layout()
    plt.show()


def get_similarity_matrix(scores, names):
    # Pre-process names to extract 'first' and 'second' components
    processed_names = [
        ("_".join(name.split("_")[:2]), "_".join(name.split("_")[-2:]).split(".")[0])
        for name in names
    ]

    unique_subs = sorted(set(["_".join(name.split("_")[:2]) for name in names]))

    mat = []

    for sub1 in unique_subs:
        row = []
        for sub2 in unique_subs:
            found = False
            for (first, second), score in zip(processed_names, scores):
                if sub1 == first and sub2 == second:
                    row.append(score)
                    found = True
                    break
            if not found:
                row.append(0)
        mat.append(row)
        print(sub1)

    return mat, unique_subs


def upscale_images(path="data/png"):
    if f.endswith(".png"):
        img = Image.open(path)

        # Get the size of the original image
        width, height = img.size

        # Upscale the image by 2x
        new_size = (width * 2, height * 2)
        upscaled_img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save the upscaled image
        upscaled_img.save(path)  # Replace with your desired output path


def plot_similarity_matrix(similarity_matrix, unique_names):
    """
    Plot the similarity matrix as a heatmap.

    Args:
    similarity_matrix (ndarray): A 2D numpy array representing the similarity matrix.
    ns1, ns2 (list): List of names used for row column labels.
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Bozorth3 Matching Points")

    skipped_names = [name if i % 3 == 0 else "" for i, name in enumerate(unique_names)]

    # Set the x and y ticks to the subject names
    plt.xticks(
        ticks=np.arange(len(unique_names)),
        labels=skipped_names,
        rotation=90,
        fontsize=8,
    )
    plt.yticks(ticks=np.arange(len(unique_names)), labels=skipped_names, fontsize=8)
    plt.title("Similarity Matrix (Bozorth3 Matching Points)", fontsize=35)
    plt.tight_layout()
    plt.show()


def get_and_plot_nfiq_scores(path="data/png"):
    subs = []
    scores = []

    for fprint in os.listdir(path):
        if not fprint.endswith(".png"):
            continue
        subs.append(fprint)

        score = sb.run(f"nfiq {path}/{fprint}", stdout=sb.PIPE)
        scores.append(int(score.stdout.decode("utf-8")))

    print(scores)
    plt.plot(scores, label="nfiq fingerprint quality")
    plt.xlabel("Fingerprint ID")
    plt.ylabel("Quality (1 = best, 5 = worst)")
    plt.ylim(1, 5)
    plt.legend()
    print(len(scores))
    plt.show()


def get_threshold():
    return 50  # Based off the chart


def get_thresholding_score(threshold, scores, names):
    truths, preds = [], []
    for score, sample in zip(scores, names):
        name1 = "_".join(sample.split("_")[:2])
        name2 = "_".join(sample.split("_")[-2:]).split(".")[0]

        # Do not compare same samples
        if name1 == name2:
            continue

        # Same subject
        if name1.split("_")[0] == name2.split("_")[0]:
            truths.append(1)  # True
        else:
            truths.append(0)

        preds.append(score > threshold)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro")

    n_matches = 0
    for p, t in zip(preds, truths):
        if p == t:
            n_matches += 1
    # print("Acc:", n_matches / len(preds))

    return acc, f1


import os
import subprocess as sb
from PIL import Image


def png_to_wsq(path="data/png"):
    files = os.listdir(path)

    for f in files:
        if f.endswith(".png"):
            # Open image to get its size
            f_base = f.replace(".png", "")

            # Generate commands using the image size
            cmd = "bash png2wsq_example.sh {}".format(f"{path}/{f_base}")
            sb.run(cmd)


def write_wsq_for_pcasys(path, file_type):
    for f in os.listdir(path):
        if f.endswith(file_type):
            with open("paths_to_your_fingerprints_for_pcasys.txt", "a") as fp:
                fp.write(f"{path}/{f} A\n")


def get_print_group(path) -> str:
    outfile = open("pcasys_final.out", "r")
    group = ""
    filename = path.split("/")[-1]
    for line in outfile:
        if filename in line.split(" ")[0]:
            # print(line.split("_")[0])
            group = line.split(" ")[5].replace(",", "")

    return group


def get_fingerprint_groups(path="data/png/"):
    groups = {"A": [], "L": [], "R": [], "S": [], "T": [], "W": []}

    for f in os.listdir(path):
        if not f.endswith(".wsq"):
            continue
        group = get_print_group(f"{path}/{f}")
        groups[group].append(f)

    return groups


def get_group_thresholding_score(threshold, scores, names1, names2):
    truths, preds = [], []
    for score, name1, name2 in zip(scores, names1, names2):
        if name1 == name2:
            continue

        # Same subject
        if name1.split("_")[0] == name2.split("_")[0]:
            truths.append(1)  # True
        else:
            truths.append(0)

        preds.append(score > threshold)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro")

    n_matches = 0
    for p, t in zip(preds, truths):
        if p == t:
            n_matches += 1
    print("Acc:", n_matches / len(preds))

    return acc, f1


def task_1():
    # Plot scores
    impostors, genuine, original = get_scores("data/matches/")
    imp, imp_s = impostors
    gen, gen_s = genuine

    plot_bozorth3_comparisons(imp, gen, imp_s, gen_s)


def task_2():
    # Plot similarity matrix
    impostors, genuine, original = get_scores("data/matches/")
    original_scores, original_names = original

    matrix, unique_names = get_similarity_matrix(original_scores, original_names)
    plot_similarity_matrix(matrix, unique_names)


def task_3():
    get_and_plot_nfiq_scores()


def task_4():
    # Determine best threshold + classification
    impostors, genuine, original = get_scores("data/matches/")

    imp, imp_s = impostors
    gen, gen_s = genuine
    original_scores, original_names = original  # For example 1_1_vs_1_2.txt

    thresh = get_threshold()

    max_acc, max_f1, best_thresh = 0, 0, 0
    thresholds = np.arange(45, 100, 1)
    for thresh in thresholds:
        acc, f1 = get_thresholding_score(thresh, original_scores, original_names)
        if acc > max_acc and f1 > max_f1:
            print(f1, max_f1)
            max_acc = acc
            max_f1 = f1
            best_thresh = thresh
        print(thresh, acc, f1)
        print()

    print("Best thresh: ", best_thresh)
    print("Best acc: ", max_acc)
    print("Best f1: ", max_f1)


def task_5():
    # Convert to wsq
    # png_to_wsq("data/png")

    # write_wsq_for_pcasys("data/png", "wsq")

    # Get groups
    groups = get_fingerprint_groups()

    # Get scores
    impostors, genuine, original = get_scores("data/matches/")
    original_scores, original_names = original

    final_group_scores = {"A": [], "L": [], "R": [], "S": [], "T": [], "W": []}

    best_group_scores = []
    thresholds = np.arange(0, 60, 1)

    # Do group thresholding
    for group, group_names in groups.items():  # List of unique names.wsq
        if len(group_names) == 0:
            continue
        original_names_first_name = [
            "_".join(sample.split("_")[:2]).split(".")[0] for sample in original_names
        ]
        original_names_second_name = [
            "_".join(sample.split("_")[-2:]).split(".")[0] for sample in original_names
        ]
        group_scores, corr_names1, corr_names2 = [], [], []

        g_names = [n.split(".")[0] for n in group_names]

        # Determine group scores
        for n1 in g_names:
            for n2 in g_names:
                if n1 == n2:
                    continue

                for i, (o1, o2) in enumerate(
                    zip(original_names_first_name, original_names_second_name)
                ):
                    if n1 == o1 and n2 == o2:
                        group_scores.append(original_scores[i])
                        corr_names1.append(n1)
                        corr_names2.append(n2)

        # print(group, group_scores, corr_names1, corr_names2)
        # exit()
        # Determine best threshold + classification
        max_acc, max_f1, best_thresh = 0, 0, 0
        for thresh in thresholds:
            acc, f1 = get_group_thresholding_score(
                thresh, group_scores, corr_names1, corr_names2
            )

            if acc > max_acc and f1 > max_f1:
                print(f1, max_f1)
                max_acc = acc
                max_f1 = f1
                best_thresh = thresh
            # print(thresh, acc, f1)
            # print()

            final_group_scores[group].append((thresh, acc, f1))

        print("Group:", group)
        print("Best thresh: ", best_thresh)
        print("Best acc: ", max_acc)
        print("Best f1: ", max_f1)

        best_group_scores.append((group, best_thresh, max_acc, max_f1))

    print(best_group_scores)

    group_thresholds = [i[1] for i in best_group_scores]
    group_accs = [i[2] for i in best_group_scores]
    group_f1s = [i[3] for i in best_group_scores]
    print()
    print(
        "Average accuracy: ",
        np.mean(group_accs),
        np.std(group_accs),
        "\nAverage f1: ",
        np.mean(group_f1s),
        np.std(group_f1s),
        "\nAverage thresh:",
        np.mean(group_thresholds),
        np.std(group_thresholds),
    )
    print(np.mean(original_scores), np.mean(group_scores))

    for k, v in groups.items():
        print(k, len(v))


if __name__ == "__main__":
    try:
        os.mkdir("data")
        os.mkdir("data/bmp")
        os.mkdir("data/png")
        os.mkdir("data/feats")
        os.mkdir("data/matches")
    except:
        pass

    subjects = get_subjects("data/bmp/")

    skip = True
    # Convert
    files = os.listdir("./data/bmp/")
    for f in files:
        convert_to_grayscale(f"./data/bmp/{f}")
        upscale_images()
    if not skip:
        # Get minutae
        files = os.listdir("./data/png")
        for f in files:
            get_features(f"./data/png/{f}")

        # Get number of matching points
        files = os.listdir("./data/feats")
        files = [f for f in files if ".xyt" in f]
        print(len(files), (len(files) - 1) ** 2)
        for i, f1 in enumerate(files):
            for f2 in files[i:]:
                if f1 == f2:
                    continue

                print("Bozorth3 scoring...", f1, f2)
                prefix = "./data/feats"

                run_bozorth3(f"{prefix}/{f1}", f"{prefix}/{f2}")

        task_1()

        task_2()

        task_3()

        task_4()

        task_5()
