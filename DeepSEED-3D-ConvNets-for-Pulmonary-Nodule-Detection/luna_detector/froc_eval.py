def common_test(self, thr, iou_threshold):
    tn = 0
    tp = 0
    n = 0
    p = 0
    fp_bbox = 0
    total_pred_len = 0
    dices = []
    target_volumes = []
    output_volumes = []
    for output, target in zip(self.outputs, self.targets):

        pred = self.gp(output, thr)
        true = self.gp(target, 0.8)
        current_dice = 0
        if len(true) > 0:
            p += 1
            correct_positive = 0
            current_dices = []
            for bbox in pred:
                cdice = dice(bbox[1:], true[0][1:])
                if cdice > iou_threshold:
                    correct_positive = 1
                else:
                    fp_bbox += 1
                # print('dice: {}'.format(cdice))
                current_dices.append([bbox[4], dice(bbox[1:], true[0][1:])])
            if len(pred) == 0:
                current_dices.append([-1, 0])
            current_dices = np.array(current_dices)
            if len(pred) > 0:
                tp += 1
            max_dice_idx = np.argmax(current_dices[:, 1])
            current_dice = current_dices[max_dice_idx][1]
            r = current_dices[max_dice_idx][0]
            dices.append(current_dice)
            target_volumes.append(true[0][4])
            output_volumes.append(r)
            total_pred_len += len(pred)
        else:
            n += 1
            if len(pred) == 0:
                tn += 1

        print('pred: {}'.format(pred))
        print('true: {}'.format(true))
        print(tp, tn, p, n, fp_bbox, current_dice)
    return [tp, tn, p, n, fp_bbox, total_pred_len], dices
