from learning.utils import *
from learning.prediction import nn_predict_dom
import math




def update_dom_nn_classifier(net, x, y, rel_map, dom, problem,
                             max_adjust_epochs=50, batch_size=32, lr=0.001,
                             acc_thr=0.9, weight_decay=0.00001):
    device = 'cuda'
    max_window_size = 11 * problem.n_var + 24
    n = len(x)
  #  start = get_start_pos(n, max_window_size)

    new_data = prepare_new_dom_data(x,y, rel_map, dom, n - 5, start=0, data_kind='tensor', device=device)
    labels, _ = nn_predict_dom(new_data[:, :-1], net)

    acc0, acc1, acc2 = get_accuracy(new_data[:, -1], labels)
    min_acc = min(acc0, acc1, acc2)

 #   print("Estimated accuracy for each class: ", acc0, acc1, acc2)

    if min_acc >= acc_thr:
        return

    data = prepare_dom_data(x,y, rel_map, dom, start=0, data_kind='tensor', device=device)

    weight = compute_class_weight(data[:, -1])
    if weight is None:
        return

    weight = torch.tensor(weight, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    adjust_epochs = max_adjust_epochs * ((acc_thr - min_acc) / acc_thr)

    adjust_epochs = math.ceil(adjust_epochs)
    train_nn(data, load_batched_dom_data, net, criterion, optimizer, batch_size, adjust_epochs)










