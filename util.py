from typing import Optional, List, Literal, Tuple
import pathlib
import json
import numpy as np
import pandas as pd
import cv2
import torch
import tqdm
import time
import matplotlib.pyplot as plt
import PySimpleGUI as gui

# general
def strtobool(val):
    """
    adapted from
    https://stackoverflow.com/questions/42248342/yes-no-prompt-in-python3-using-strtobool

    Convert a string representation of truth to true (1) or false (0).
    Raises ValueError if 'val' is anything else other than those defined as true or false.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return bool(1)
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return bool(0)
    else:
        raise ValueError("invalid truth value %r" % (val,))

# pandas
def df_to_string(df: pd.DataFrame, n: int = 0):
    """
    Converts a dataframe to string.

    Args:
        n:
            number of indentations for the whole dataframe as a table
    """
    indent = " " * n
    x = indent + df.to_string().replace("\n", "\n" + indent)
    return x

# directory related
def move_files(files: List[pathlib.Path], new_dir: str):
    """
    move files from one directory to new dir
    """
    dir = pathlib.Path(new_dir)
    if not dir.exists() or not dir.is_dir():
        raise UserWarning(f"{dir = } does not exist.")
    else:
        for x in files:
            if x.exists() and x.is_file():
                x.rename(dir/x.name)

def search_files(path: str, file_name: str, exts: Optional[List[str]],
                recursive: bool, file_name_match_type: Literal["any", "exact"]):
    """
    Searches through subdir. specified in `path` (if recursive set to True) for files whose name contain/ is exact match to `file_name`.
    
    Args:
        path: path to search for file
        file_name: file_name to filter (without extension)
        exts: extensions of file interested in. if None, assumes an ext already exist in filename and use it.
        recursive: if True, searches subdirectories too
        file_name_match_typ:
            "any": matches with any files that contains the strin in `file_name`
            "except": only matches files that has name == `file_name`

    Returns:
        list of filepaths for file whose name contains `file_name`

    # Example usage
    files = util.search_files(<directory to search for file>, 
                            '20231208_121138', exts=["txt", "png"],
                            recursive=False, file_name_match_type="exact")
    for file in files:
        print(file)
    """ 
    EXACT = "exact"
    ANY = "any"

    def search(recursive: bool, files: list, path: pathlib.Path, pattern: str):
        if recursive:
            files.extend(list(path.glob(f"**/{pattern}")))
        else:
            files.extend(list(path.glob(pattern)))

    assert file_name_match_type in [EXACT, ANY], "Invalid exts used."
    message = (f"{exts = }\n"
               f"Confirm using {recursive = }\n?"
               f"{file_name_match_type = }")
    if exts is None:
        if file_name_match_type != EXACT:
            file_name_match_type = EXACT
            message += (f"\n\nfile_name_match_type changed to \"{file_name_match_type}\" due to {exts = }")

    proceed = strtobool(gui.popup(message, title = f"{__name__}", button_type=1,
                                  keep_on_top=True))

    # Create a Path object
    path = pathlib.Path(path)

    # Check if the path exists
    if not path.exists():
        print(f"The path {path} does not exist.")
        return []

    # Prepare the file name pattern
    files = []
    if proceed:
        if isinstance(exts, list):
            for ext in exts:
                if file_name_match_type == ANY:
                    pattern = f"*{file_name}*.{ext}"
                elif file_name_match_type == EXACT:
                    pattern = f"{file_name}.{ext}"
                search(recursive, files, path, pattern)
        elif exts is None:
            pattern = f"{file_name}"
            search(recursive, files, path, pattern)
        else:
            raise UserWarning(f"Invalid {exts = }")

        # Return the list of file paths
        return [str(file.resolve()) for file in files]
    else:
        raise UserWarning(f"User selected {proceed = }")

def compare_file_name(path1: str, path2: str, ext: List[str], recursive: Tuple[bool, bool]):
    """
    Navigates the subdirectories of path1 and path2 and compare filenames of files with extension. 
    
    Returns: 
        file names  that are:
        - in path1 and not in path2
        - in path2 and not in path1
        - present in both paths

    Example usage: 
        # path1 = r"<my path on computer>"
        # path2 = r'<my path on computer>'

        in_path1_not_path2, in_path2_not_path1, in_both_paths = compare_files(path1, path2)
        if isinstance(in_path1_not_path2, str):
            print(in_path1_not_path2)
        elif isinstance(in_path2_not_path1, str):
            print(in_path2_not_path1)
        else:
            print(f"{len(in_path1_not_path2)} Files in path1 but not in path2:\n", in_path1_not_path2)
            print(f"{len(in_path2_not_path1)} Files in path2 but not in path1:\n", in_path2_not_path1)
            print(f"{len(in_both_paths)} Files in both paths:\n", in_both_paths)
    """
    assert isinstance(recursive, tuple) and len(recursive)==2 and all([isinstance(x, bool) for x in recursive]), f"Invalid recursive setting. {recursive = }"
    msg = (f"Confirm using\n"
           f"{ext = }\n"
           f"{recursive = }")
    proceed = strtobool(gui.popup(msg, title = f"{__name__}", button_type=1,
                                  keep_on_top=True))

    # Check if the directories exist
    if not pathlib.Path(path1).is_dir():
        return f"Directory {path1} does not exist.", None, None
    if not pathlib.Path(path2).is_dir():
        return None, f"Directory {path2} does not exist.", None
    
    if proceed:
        files_in_path1, files_in_path2 = set(), set()

        for extension in ext:
            # Get the set of all .jpg and .png files in path1
            # files_in_path1 = {f.name for f in Path(path1).glob('*.[jp][np]g')} # this searches through dir only
            # files_in_path1 = {f.name for f in pathlib.Path(path1).glob(f'**/*.{ext}')} # this searches through subfolder too
            if recursive[0]:
                files_in_path1.update({f.name for f in pathlib.Path(path1).rglob(f'*.{extension}')})
            else:
                files_in_path1.update({f.name for f in pathlib.Path(path1).glob(f'*.{extension}')})
        
            # Get the set of all .jpg and .png files in path2
            # files_in_path2 = {f.name for f in Path(path2).glob('*.[jp][np]g')} # this searches through dir only
            if recursive[1]:
                files_in_path2.update({f.name for f in pathlib.Path(path2).rglob(f'*.{extension}')}) # this searches through subfolder too
            else:
                files_in_path2.update({f.name for f in pathlib.Path(path2).glob(f'*.{extension}')})

        # Files in path1 but not in path2
        in_path1_not_path2 = list(files_in_path1 - files_in_path2)

        # Files in path2 but not in path1
        in_path2_not_path1 = list(files_in_path2 - files_in_path1)

        in_both_paths = list(files_in_path1 & files_in_path2)

        return in_path1_not_path2, in_path2_not_path1, in_both_paths
    else:
        raise UserWarning(f"User selected {proceed = }")

# ultralytics
class UltralyticsUtils:
    def obj_det(self, 
                save_dir: str, 
                img_folder_path: str, 
                yolov5_repo_path: str, 
                model_path: str, 
                model_conf: float,
                inf_sz: int = 480,
                output_sz: Tuple[int, int] = (1000, 769),
                save_img: bool = True,
                save_pandas: bool = False,
                incl: List[str] = list(),
                excl: List[str] = list(),):
        """
        Performs obj det with yolov5 from ultralytics and save images with result annotated + pandas result.
        Args:
            save_dir: str
                which directory to save results to 
            img_folder_path: str
                folder containing images to inference on
            yolov5_repo_path: str
                folder containing yolov5 cloned repo
            model_path: str
                path to model.pt file
            model_conf: float
                if >0, sets model.conf to model_conf
            inf_sz: int
                size to perform inference using model
            output_sz:
                image output sz
            save_pandas:
                whether to save pandas result
            save_img:
                whether to save img with resutles rendered
            incl:
                list of img (name+extension) to perform inference on
            excl:
                list of img (name+extension) to not perform inference on
        """
        start_time = time.perf_counter()
        save_dir = pathlib.Path(save_dir) 
        img_folder_path = pathlib.Path(img_folder_path)

        if not img_folder_path.exists() or not img_folder_path.is_dir():
            raise UserWarning("issue with img_folder_path")
        
        images = list(img_folder_path.glob('*.[jp][np][g]*'))
        if not images:
            raise UserWarning("no images found.")

        save_dir.mkdir(parents=True, exist_ok=True)

        if any(p.is_file() for p in save_dir.iterdir()):
            raise UserWarning("a file exist, make sure the images are no longer required/ saved at another dir/ subdir")

        if incl:
            images = [img for img in images if img.name in incl]
        if excl:
            images = [img for img in images if img.name not in excl]

        total_images = len(images)
        if total_images>0:
            model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path,
                source='local', force_reload=True)
            if model_conf>0:
                model.conf = model_conf
        time.sleep(0.01)
        progress_bar = tqdm.tqdm(total=total_images)
        for img in images:
            results = model(img, size=inf_sz)
            results.render()

            if save_pandas:
                df = results.pandas().xyxy[0]
                txt_file = str(save_dir/f'{img.stem}.txt')
                with open(txt_file, "w") as f:
                    f.write(df_to_string(df))
            for im in results.ims:
                if save_img:
                    cv2.imwrite(str(save_dir/f"{img.stem}.png"), 
                                cv2.resize(im, output_sz)[:,:,::-1])
            progress_bar.update()

        print(f"completed in {time.perf_counter() - start_time} s.")
        progress_bar.close()

# Plotly
class PlotlyUtils:
    def plot(self):
        import plotly.graph_objects as go
        xlim = [-4, 4]

        # two vectors in R3
        v1 = np.array([ 3,5,1 ])
        v2 = np.array([ 0,2,2 ])

        scalars = np.random.uniform(low=xlim[0],high=xlim[1],size=(100,2))

        points = np.outer(scalars[:,0],v1)+np.outer(scalars[:,1],v2)

        # draw the dots in the plane
        fig = go.Figure(data=[go.Scatter3d(x=points[:,0], 
                                        y=points[:,1], 
                                        z=points[:,2], 
                                        mode='markers', 
                                        marker=dict(size=6,color='black'))])
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
        # plt.savefig('Figure_03_07b.png',dpi=300)
        fig.show()

# Matplotlib
class MatplotlibUtils:
    """
    the book "Practical linear algebra for data science: from core concepts to applications using Python" 
    (ISBN: 978-1-09-812061-0) uses the following line in ipynb to setup for Orielly
        import matplotlib_inline.backend_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # print figures in svg format
        plt.rcParams.update({'font.size':14}) # set global font size
    """

    """
    to test , assuming we don't have any other attributes or methods other than `plot_<>` and `plot`, we can do 
        for i in range(999): # specify depth
            try:
                if not i:
                    x.__getattribute__("plot")()
                else:
                    x.__getattribute__(f"plot{i+1}")()
            except AttributeError as e:
                print(f"{e = }")
                break
    """

    def plot(self):
        # create a vector
        v = np.array([-1,2])

        # plot that vector (and a dot for the tail)
        plt.arrow(0,0,v[0],v[1],head_width=.5,width=.1)
        plt.plot(0,0,'ko',markerfacecolor='k',markersize=7)

        # add axis lines
        plt.plot([-3,3],[0,0],'--',color=[.8,.8,.8],zorder=-1) # zorder allows layering like photoshop layers
        plt.plot([0,0],[-3,3],'--',color=[.8,.8,.8],zorder=-1)

        # make the plot look nicer
        plt.axis('square')
        plt.axis([-3,3,-3,3])
        plt.xlabel('$v_0$')
        plt.ylabel('$v_1$')
        plt.title('Vector v in standard position')
        plt.show()

    def plot2(self):
        # A range of starting positions
        
        # create a vector
        v = np.array([-1,2])
        startPos = [
                    [0,0],
                    [-1,-1],
                    [1.5,-2]
                    ]


        # create a new figure
        fig = plt.figure(figsize=(6,6))

        for s in startPos:

            # plot that vector (and a dot for the tail)
            # note that plt.arrow automatically adds an offset to the third/fourth inputs
            plt.arrow(s[0],s[1],v[0],v[1],head_width=.5,width=.1,color='black')
            plt.plot(s[0],s[1],'ko',markerfacecolor='k',markersize=7)

            # indicate the vector in its standard position
            if s==[0,0]:
                plt.text(v[0]+.1,v[1]+.2,'"Standard pos."')


        # add axis lines
        plt.plot([-3,3],[0,0],'--',color=[.8,.8,.8],zorder=-1)
        plt.plot([0,0],[-3,3],'--',color=[.8,.8,.8],zorder=-1)

        # make the plot look nicer
        plt.axis('square')
        plt.axis([-3,3,-3,3])
        plt.xlabel('$v_0$')
        plt.ylabel('$v_1$')
        plt.title('Vector $\mathbf{v}$ in various locations')
        # plt.savefig('Figure_02_01.png',dpi=300) # write out the fig to a file
        plt.show()

    def plot3(self):
        # a scalar
        s = 3.5

        # a vector
        b = np.array([3,4])

        # plot
        plt.plot([0,b[0]],[0,b[1]],'m--',linewidth=3,label='b')
        plt.plot([0,s*b[0]],[0,s*b[1]],'k:',linewidth=3,label='sb')

        plt.grid()
        plt.axis('square')
        plt.axis([-6,6,-6,6])
        plt.legend()
        plt.show()

    def plot4(self):
        # Effects of different scalars

        # a list of scalars:
        scalars = [ 1, 2, 1/3, 0, -2/3 ]

        baseVector = np.array([ .75,1 ])

        # create a figure
        fig,axs = plt.subplots(1,len(scalars),figsize=(12,3))
        i = 0 # axis counter

        for s in scalars:

            # compute the scaled vector
            v = s*baseVector

            # plot it
            axs[i].arrow(0,0,baseVector[0],baseVector[1],head_width=.3,width=.1,color='k',length_includes_head=True)
            axs[i].arrow(.1,0,v[0],v[1],head_width=.3,width=.1,color=[.75,.75,.75],length_includes_head=True)
            axs[i].grid(linestyle='--')
            axs[i].axis('square')
            axs[i].axis([-2.5,2.5,-2.5,2.5])
            axs[i].set(xticks=np.arange(-2,3), yticks=np.arange(-2,3))
            axs[i].set_title(f'$\sigma$ = {s:.2f}')
            i+=1 # update axis counter

        plt.tight_layout()
        # plt.savefig('Figure_02_03.png',dpi=300)
        plt.show()

    def plot5(self):
        v = np.array([1, 2])
        w = np.array([4, -6])
        u = v+w

        plt.figure(figsize=(6,6))

        a1 = plt.arrow(0,0,v[0],v[1],head_width=.3,width=.1,color='k',length_includes_head=True)
        a2 = plt.arrow(v[0],v[1],w[0],w[1],head_width=.3,width=.1,color=[.5,.5,.5],length_includes_head=True)
        a3 = plt.arrow(0,0,u[0],u[1],head_width=.3,width=.1,color=[.8,.8,.8],length_includes_head=True)


        # make the plot look a bit nicer
        plt.grid(linestyle='--',linewidth=.5)
        plt.axis('square')
        plt.axis([-6,6,-6,6])
        plt.legend([a1,a2,a3],['v','w','v+w'])
        plt.title('Vectors $\mathbf{v}$, $\mathbf{w}$, and $\mathbf{v+w}$')
        # plt.savefig('Figure_02_02a.png',dpi=300) # write out the fig to a file
        plt.show()

    def plot6(self):
        # the vectors a and b
        a = np.array([1,2])
        b = np.array([1.5,.5])

        # compute beta
        beta = np.dot(a,b) / np.dot(a,a)

        # compute the projection vector (not explicitly used in the plot)
        projvect = b - beta*a

        # draw the figure
        plt.figure(figsize=(4,4))

        # vectors
        plt.arrow(0,0,a[0],a[1],head_width=.2,width=.02,color='k',length_includes_head=True)
        plt.arrow(0,0,b[0],b[1],head_width=.2,width=.02,color='k',length_includes_head=True)

        # projection vector
        plt.plot([b[0],beta*a[0]],[b[1],beta*a[1]],'k--')

        # projection on a
        plt.plot(beta*a[0],beta*a[1],'ko',markerfacecolor='w',markersize=13)

        # make the plot look nicer
        plt.plot([-1,2.5],[0,0],'--',color='gray',linewidth=.5)
        plt.plot([0,0],[-1,2.5],'--',color='gray',linewidth=.5)

        # add labels
        plt.text(a[0]+.1,a[1],'a',fontweight='bold',fontsize=18)
        plt.text(b[0],b[1]-.3,'b',fontweight='bold',fontsize=18)
        plt.text(beta*a[0]-.35,beta*a[1],r'$\beta$',fontweight='bold',fontsize=18)
        plt.text((b[0]+beta*a[0])/2,(b[1]+beta*a[1])/2+.1,r'(b-$\beta$a)',fontweight='bold',fontsize=18)

        # some finishing touches
        plt.axis('square')
        plt.axis([-1,2.5,-1,2.5])
        plt.show()

    def plot7(self):
        # generate random R2 vectors (note: no orientation here! we don't need it for this exercise)
        t = np.random.randn(2)
        r = np.random.randn(2)

        print(f"{t = }\n{r = }")

        # the decomposition
        t_para = r * (np.dot(t,r) / np.dot(r,r))
        t_perp = t - t_para

        # confirm orthogonality (dot product must be zero!)
        assert np.dot(t_para, t_perp)<1e-9, f"{np.dot(t_para, t_perp)}"
        # Note about this result: Due to numerical precision errors, 
        #   you might get a result of something like 10^-17, which can be interpretd as zero.

        # draw them!
        plt.figure(figsize=(4,4))

        # draw main vectors
        plt.plot([0,t[0]],[0,t[1]],color='k',linewidth=3,label=r'$\mathbf{t}$')
        plt.plot([0,r[0]],[0,r[1]],color=[.7,.7,.7],linewidth=3,label=r'$\mathbf{r}$')

        # draw decomposed vector components
        plt.plot([0,t_para[0]],[0,t_para[1]],'k--',linewidth=3,label=r'$\mathbf{t}_{\|}$')
        plt.plot([0,t_perp[0]],[0,t_perp[1]],'k:',linewidth=3,label=r'$\mathbf{t}_{\perp}$')

        plt.axis('equal')
        plt.legend()
        # plt.savefig('Figure_02_08.png',dpi=300)
        plt.show()

    def plot8(self):
        # points (in Cartesian coordinates)
        p = (3,1)
        q = (-6,2)

        plt.figure(figsize=(6,6))

        # draw points
        plt.plot(p[0], p[1],'ko',markerfacecolor='k',markersize=10,label='Point p')
        plt.plot(q[0], q[1],'ks',markerfacecolor='k',markersize=10,label='Point q')

        # draw basis vectors
        # note syntax: plt.plot(<list of x>, <list of y>, <args/kwargs>)
        plt.plot([0,0],[0,1],'k',linewidth=3, label='Basis S')
        plt.plot([0,1],[0,0],'k--',linewidth=3)

        plt.axis('square')
        plt.grid(linestyle='--',color=[.8,.8,.8])
        plt.xlim([-7,7])
        plt.ylim([-7,7])
        plt.legend()
        # plt.savefig('Figure_03_04.png',dpi=300)
        plt.show()

    def plot9(self):
        A = np.array([1, 3])

        xlim = [-4, 4]
        scalars = np.random.uniform(low=xlim[0], high=xlim[1], size=100)
        print(f"{scalars = }")

        output = np.outer(scalars, A)
        print(f"{output.shape = }")

        plt.figure(figsize=(6,6))
        plt.scatter(output[:,0], output[:,1], color="k", marker="o")
        plt.xlim(xlim)
        plt.ylim(xlim)
        plt.grid()
        plt.text(-4.5, 4.5, "A", fontweight="bold", fontsize=18)
        plt.show()

# json
class CustomJSONEncoder(json.JSONEncoder):
    """
    Generated with GPT-4. A fix for encoding a dict that may contain numpy_bool and dataframe.
    orient = "split" is to counter edge where if index is not unique, orient = "index" or "columns" causes ValueError

    Example use case:
        data = <a dictionary containing numpy.bool_ and pandas dataframe>

        with open(folder/file_name, "w") as op:
            json.dump(data, op, cls=CustomJSONEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_json(orient="split" , indent=1)
        return json.JSONEncoder.default(self, obj)