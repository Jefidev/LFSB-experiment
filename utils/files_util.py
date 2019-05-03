from os import makedirs, path


def make_dirs(to_create):
    if not path.exists(to_create):
        makedirs(to_create)


def load_class(classpath):
    elems = classpath.split(".")
    classname = elems.pop()
    classfile = ".".join(elems)

    mod = __import__(classfile, fromlist=[classname])
    return getattr(mod, classname)
