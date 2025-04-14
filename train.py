import torch
import argparse

from torchvision import transforms
from torchvision.ops import SqueezeExcitation

# set random seed
torch.manual_seed(0)

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="uskc", help="Dataset to use", choices=["uskc", "uskc_defect"])
parser.add_argument("--root_path", type=str, default="D:/Research/Dataset/USK-Coffee", help="Root path of the dataset")
parser.add_argument("--rand_aug", type=bool, default=False, help="Use RandAugment")
parser.add_argument("--add_se", type=bool, default=False, help="Add squeeze and excitation block")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
parser.add_argument("--ckpt_save", type=str, default="best_model_resnet18.pth", help="Model path to save")
parser.add_argument("--model", type=str, default="resnet18", help="Model to use", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vitb", "swinb", "efficientnetb7"])
parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
parser.add_argument("--scheduler", type=str, default="constant", help="Scheduler to use", choices=["constant", "cosine", "step"])
args = parser.parse_args()

if __name__ == "__main__":
    # Define transforms conditionally
    train_transform = []

    if args.rand_aug:
        # Apply RandAugment on the original 256x256 image before cropping
        train_transform.append(transforms.RandAugment(num_ops=2, magnitude=9))

    # Always apply CenterCrop, Random horizontal flip, ToTensor, and Normalization
    train_transform.extend([
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose(train_transform)
    
    # Test transform
    test_transform = transforms.Compose([
        # transforms.Resize((256, 256)), # image size is already 256x256
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # define the dataset
    if args.dataset == "uskc":
        from USKCoffeeDataset import USKCoffeeDataset as CoffeeDataset
        num_classes = 4
    elif args.dataset == "uskc_defect":
        from USKCoffeeDataset import USKCoffeeDatasetDefect as CoffeeDataset
        num_classes = 2
    
    train_dataset = CoffeeDataset(root_path=args.root_path, phase="train", transform=train_transform)
    val_dataset = CoffeeDataset(root_path=args.root_path, phase="val", transform=test_transform)
    test_dataset = CoffeeDataset(root_path=args.root_path, phase="test", transform=test_transform)
    
    # define the dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=args.num_workers)
    
    # define the model
    if args.model == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        if args.pretrained:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, num_classes) 
    elif args.model == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        if args.pretrained:
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = resnet34(weights=None)
        model.fc = torch.nn.Linear(512, num_classes)
    elif args.model == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        if args.pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = resnet50(weights=None)
        model.fc = torch.nn.Linear(2048, num_classes)
    # TODO: Smua RESNET, 101, 152, ViTB, SwinB, EfficientNetB7
    elif args.model == "resnet101":
        from torchvision.models import resnet101, ResNet101_Weights
        if args.pretrained:
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            model = resnet101(weights=None)
        model.fc = torch.nn.Linear(2048, num_classes)
    elif args.model == "resnet152":
        from torchvision.models import resnet152, ResNet152_Weights
        if args.pretrained:
            model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        else:
            model = resnet152(weights=None)
        model.fc = torch.nn.Linear(2048, num_classes)
    elif args.model == "vitb":
        from torchvision.models import vit_b_16
        if args.pretrained:
            model = vit_b_16(weights="DEFAULT")
        else:
            model = vit_b_16(weights=None)
        model.heads.head = torch.nn.Linear(768, num_classes)
    elif args.model == "swinb":
        from torchvision.models import swin_b
        if args.pretrained:
            model = swin_b(weights="DEFAULT")
        else:
            model = swin_b(weights=None)
        model.head = torch.nn.Linear(1024, num_classes)
    elif args.model == "efficientnetb7":
        from torchvision.models import efficientnet_b7
        if args.pretrained:
            model = efficientnet_b7(weights="DEFAULT")
        else:
            model = efficientnet_b7(weights=None)
        model.classifier[1] = torch.nn.Linear(2560, num_classes)
    else:
        raise ValueError("Model not supported")
    
    # default init for resnet for training from scratch
    if not args.pretrained:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    if args.add_se:
        if args.model == "resnet18" or args.model == "resnet34":
            # add squeeze and excitation block
            model.layer1[0].se = SqueezeExcitation(64, 16)
            model.layer2[0].se = SqueezeExcitation(128, 32)
            model.layer3[0].se = SqueezeExcitation(256, 64)
            model.layer4[0].se = SqueezeExcitation(512, 128)
        elif args.model == "resnet50":
            model.layer1[0].se = SqueezeExcitation(256, 64)
            model.layer2[0].se = SqueezeExcitation(512, 128)
            model.layer3[0].se = SqueezeExcitation(1024, 256)
            model.layer4[0].se = SqueezeExcitation(2048, 512)
        
        # initialize SE module weights
        for m in model.modules():
            if isinstance(m, SqueezeExcitation):
                torch.nn.init.kaiming_normal_(m.fc1.weight)
                torch.nn.init.kaiming_normal_(m.fc2.weight)
    
    # initialize the fc layer
    if args.model != "vitb" and args.model != "swinb":
        torch.nn.init.kaiming_normal_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)
    else:
        if args.model == "swinb":
            torch.nn.init.kaiming_normal_(model.head.weight)
            torch.nn.init.zeros_(model.head.bias)
        elif args.model == "vitb":
            torch.nn.init.kaiming_normal_(model.heads.head.weight)
            torch.nn.init.zeros_(model.heads.head.bias)
    
    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # use cosine annealing learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # move the model to the device
    model.to(device)
    
    # set tensorboard
    from torch.utils.tensorboard import SummaryWriter
    log_file_path = f"runs/{args.dataset}/{args.model}_{'se' if args.add_se else 'no-se'}/ep_{args.num_epochs}/lr_{args.lr}/{'randaug' if args.rand_aug else 'no-randaug'}_{args.scheduler}"
    writer = SummaryWriter(log_file_path)
    
    # training loop
    prev_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        scheduler.step()
        
        # validation loop, monitor the loss
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            val_loss = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels)
            writer.add_scalar("Loss/val", val_loss.item(), epoch)
            writer.flush()
            print(f"Validation Accuracy: {correct/total}, Validation Loss: {val_loss}")
            
            # save the best model
            if prev_loss > val_loss:
                prev_loss = val_loss
                torch.save(model.state_dict(), f"{log_file_path}/ckpt_best.pth")
                
    # load the best model
    model.load_state_dict(torch.load(args.ckpt_save))
            
    # testing loop
    model.eval()
    
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        writer.add_scalar("Loss/test", val_loss.item(), epoch)
        writer.flush()
        writer.close()
        print(f"Test Accuracy: {correct/total}")