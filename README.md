# Image Forgery Detection Tool
The forgery detection tool contained in this repository currently features forensic methods to detect the following:

- Double JPEG compression
- Copy-move forgeries
- CFA artifacts
- Noise variance inconsitencies

Please read our paper for a detailed explanation of our motivation and research when developing this tool.

## To Run:
Place any images that you wish to analyze into the **images** directory.

Navigate to the **src** directory:
```
$ cd src
```

Next, run the **detect.py** script, providing the image you wish to evaluate:
```
$ python detect.py image.jpg
```

Once finished, details on the image will be reported in the terminal. Supplemental images generated during copy-move forgery detection can be found in the output directory.
