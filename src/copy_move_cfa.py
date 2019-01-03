# Implementation derived from vasiliauskas.agnius@gmail.com

import sys
from PIL import Image, ImageFilter, ImageDraw
import operator as op
from optparse import OptionParser

def Dist(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    return (((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)))**0.5

def intersectarea(p1,p2,size):
    x1, y1 = p1
    x2, y2 = p2
    ix1, iy1 = max(x1,x2), max(y1,y2)
    ix2, iy2 = min(x1+size,x2+size), min(y1+size,y2+size)
    iarea = abs(ix2-ix1)*abs(iy2-iy1)
    if iy2 < iy1 or ix2 < ix1: iarea = 0
    return iarea

def Hausdorff_distance(clust1, clust2, forward, dir):
    if forward == None:
        return max(Hausdorff_distance(clust1,clust2,True,dir),Hausdorff_distance(clust1,clust2,False,dir))
    else:
        clstart, clend = (clust1,clust2) if forward else (clust2,clust1)
        dx, dy = dir if forward else (-dir[0],-dir[1])
        return sum([min([Dist((p1[0]+dx,p1[1]+dy),p2) for p2 in clend]) for p1 in clstart])/len(clstart)

def hassimilarcluster(ind, clusters, opt):
    item = op.itemgetter
    found = False
    tx = min(clusters[ind],key=item(0))[0]
    ty = min(clusters[ind],key=item(1))[1]
    for i, cl in enumerate(clusters):
        if i != ind:
            cx = min(cl,key=item(0))[0]
            cy = min(cl,key=item(1))[1]
            dx, dy = cx - tx, cy - ty
            specdist = Hausdorff_distance(clusters[ind],cl,None,(dx,dy))
            if specdist <= int(opt.rgsim):
                found = True
                break
    return found

def blockpoints(pix, coords, size):
    xs, ys = coords
    for x in range(xs,xs+size):
        for y in range(ys,ys+size):
            yield pix[x,y]

def colortopalette(color, palette):
    for a,b in palette:
        if color >= a and color < b:
            return b

def imagetopalette(image, palcolors):
    assert image.mode == 'L', "Only grayscale images supported !"
    pal = [(palcolors[i],palcolors[i+1]) for i in range(len(palcolors)-1)]
    image.putdata([colortopalette(c,pal) for c in list(image.getdata())])

def getparts(image, block_len, opt):
 img = image.convert('L') if image.mode != 'L' else image
 w, h = img.size
 parts = []
 # Bluring image for abandoning image details and noise.
 for n in range(int(opt.imblev)):
  img = img.filter(ImageFilter.SMOOTH_MORE)
 # Converting image to custom palette
 imagetopalette(img, [x for x in range(256) if x%int(opt.impalred) == 0])
 pix = img.load()

 for x in range(w-block_len):
  for y in range(h-block_len):
   data = list(blockpoints(pix, (x,y), block_len)) + [(x,y)]
   parts.append(data)
 parts = sorted(parts)
 return parts

def similarparts(imagparts, opt):
 dupl = []
 l = len(imagparts[0])-1

 for i in range(len(imagparts)-1):
  difs = sum(abs(x-y) for x,y in zip(imagparts[i][:l],imagparts[i+1][:l]))
  mean = float(sum(imagparts[i][:l])) / l
  dev = float(sum(abs(mean-val) for val in imagparts[i][:l])) / l
  if mean == 0: mean = .000000000001
  if dev/mean >= float(opt.blcoldev):
   if difs <= int(opt.blsim):
    if imagparts[i] not in dupl:
     dupl.append(imagparts[i])
    if imagparts[i+1] not in dupl:
     dupl.append(imagparts[i+1])

 return dupl

def clusterparts(parts, block_len, opt):
 parts = sorted(parts, key=op.itemgetter(-1))
 clusters = [[parts[0][-1]]]

 # assign all parts to clusters
 for i in range(1,len(parts)):
  x, y = parts[i][-1]

  # detect box already in cluster
  fc = []
  for k,cl in enumerate(clusters):
   for xc,yc in cl:
    ar = intersectarea((xc,yc),(x,y),block_len)
    intrat = float(ar)/(block_len*block_len)
    if intrat > float(opt.blint):
     if not fc: clusters[k].append((x,y))
     fc.append(k)
     break

  # if this is new cluster
  if not fc:
   clusters.append([(x,y)])
  else:
   # re-clustering boxes if in several clusters at once
   while len(fc) > 1:
    clusters[fc[0]] += clusters[fc[-1]]
    del clusters[fc[-1]]
    del fc[-1]

 item = op.itemgetter
 # filter out small clusters
 clusters = [clust for clust in clusters if Dist((min(clust,key=item(0))[0],min(clust,key=item(1))[1]), (max(clust,key=item(0))[0],max(clust,key=item(1))[1]))/(block_len*1.4) >= float(opt.rgsize)]

 # filter out clusters, which doesn`t have identical twin cluster
 clusters = [clust for x,clust in enumerate(clusters) if hassimilarcluster(x, clusters, opt)]

 return clusters

def marksimilar(image, clust, size, opt):
 blocks = []
 if clust:
  draw = ImageDraw.Draw(image)
  mask = Image.new('RGB', (size,size), 'cyan')
  for cl in clust:
   for x,y in cl:
   	im = image.crop((x,y,x+size,y+size))
   	im = Image.blend(im,mask,0.5)
   	blocks.append((x,y,im))
  for bl in blocks:
  	x,y,im = bl
  	image.paste(im,(x,y,x+size,y+size))
  if int(opt.imauto):
   for cl in clust:
    cx1 = min([cx for cx,cy in cl])
    cy1 = min([cy for cx,cy in cl])
    cx2 = max([cx for cx,cy in cl]) + block_len
    cy2 = max([cy for cx,cy in cl]) + block_len
    draw.rectangle([cx1,cy1,cx2,cy2],outline="magenta")
 return image

def detect(input, opt, args):
    block_len = 15
    im = Image.open(input)
    lparts = getparts(im, block_len, opt)
    dparts = similarparts(lparts, opt)
    cparts = clusterparts(dparts, block_len, opt) if int(opt.imauto) else [[elem[-1] for elem in dparts]]
    im = marksimilar(im, cparts, block_len, opt)
    out = input.split('.')[0] + '_analyzed.jpg'
    im.save(out)
    identical_regions = len(cparts) if int(opt.imauto) else 0
    print('\tCopy-move output is saved in file -', out)
    return(identical_regions)
