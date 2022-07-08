from util.visualization import draw_loss, draw_loss_epoch, draw_coco_eval

draw_loss("120E_8B_608_608_yolo_resnet18_test.txt")
draw_loss_epoch("120E_8B_608_608_yolo_resnet18_test.txt",1875)
draw_coco_eval("120E_8B_608_608_yolo_resnet18_test_eval.txt")