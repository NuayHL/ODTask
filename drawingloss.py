from util.visualization import draw_loss, draw_loss_epoch, draw_coco_eval

draw_loss("100E_8B_800_1024_Retinanet2nd_test.txt")
draw_loss_epoch("100E_8B_800_1024_Retinanet2nd_test.txt",50)
#draw_coco_eval("100E_16B_608_608_Darknet53NoFocal_widerperson_eval.txt")