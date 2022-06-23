from util.visualization import draw_loss, draw_loss_epoch, draw_coco_eval

draw_loss("100E_16B_608_608_Darknet53NoFocal.txt")
draw_loss_epoch("100E_16B_608_608_Darknet53NoFocal.txt",938)
draw_coco_eval("70E_8B_800_1024_darknet53_eval.txt")