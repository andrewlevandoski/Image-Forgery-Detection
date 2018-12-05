# Image Forgery Detection
---
Please read our paper for a detailed explanation of our motivation and research when developing this tool.

## To Run:
---
Navigate to the **src** directory:
```
cd src
```

Next, run the **detect.py** script, providing the image you wish to evaluate:

When the index is ready, you can search without a server using `search.sh`:
```
python detect.py image.jpg
```

Once finished, details on the image will be reported in the terminal. Supplemental images generated during copy-move forgery detection can be found in the output directory.
