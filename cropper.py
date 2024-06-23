from predata import crop_images, remove_ruler

crop_images('OCT', 'OCT_crops')
remove_ruler('OCT_crops', 'OCT_noruler')
