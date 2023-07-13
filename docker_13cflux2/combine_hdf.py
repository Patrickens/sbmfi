from shutil import move
import os, sys
import glob
import tables as pt

# TODO: move this stuff to fml_parser.py and somehow execute from PowerShell

def combine_hdf(file):
    for i, path in enumerate(glob.glob(f'out_{file}_*[0-9].h5')):
        othdf = pt.open_file(filename=path, mode='a')
        if i == 0:
            hdf = othdf
            hdf.root.jacobian._f_move(newname='config_0')
            hdf.root.config_0._f_move(newparent='/jacobian', createparents=True)
            continue
        coname = f'config_{i}'
        othdf.root.jacobian._g_copy(newparent=hdf.root.jacobian, newname=coname, recursive=True)
        othdf.close()
        os.remove(path=path)
    hdf.close()
    move(hdf.filename, f'out_{file}.h5')

if __name__ == "__main__":
    combine_hdf(file=sys.argv[1])
    #combine_hdf('spiro_parsed')

