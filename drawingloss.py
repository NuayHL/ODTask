from util.visualization import draw_loss, draw_loss_epoch, draw_coco_eval

draw_loss("250E_16B_608_608_yolo_resnet18_temp.txt")
draw_loss_epoch("250E_16B_608_608_yolo_resnet18_temp.txt",938)
#draw_coco_eval("120E_8B_608_608_yolo_resnet18_test_eval.txt")