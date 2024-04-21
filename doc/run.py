import json
import os
import subprocess


def build_doc(version, tag):
    os.environ["current_version"] = version
    subprocess.run("git checkout " + tag, shell=True)
    subprocess.run("git checkout master -- source/conf.py", shell=True)
    subprocess.run("git checkout master -- source/versions.json", shell=True)
    subprocess.run("make html", shell=True)


def move_dir(src, dst):
    subprocess.run(["mkdir", "-p", dst])
    subprocess.run("mv " + src + "* " + dst, shell=True)


if __name__ == "__main__":
    ## First clean dir
    subprocess.run("rm -rf build/ pages/", shell=True)

    os.environ["build_all_docs"] = str(True)
    os.environ["pages_root"] = "https://scq-cloud.github.io"

    build_doc("latest", "master")
    move_dir("./build/html/", "./pages/")

    with open("source/versions.json", "r") as json_file:
        docs = json.load(json_file)

    for version in docs:
        tag = docs[version]["tag"]
        build_doc(version, tag)
        move_dir("./build/html/", "./pages/" + version)
