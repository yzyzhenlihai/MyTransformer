from tqdm import tqdm
import torch

def train_epoch(model, loader, criterion, optimizer, scheduler, device):

    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for src, tgt_input, tgt_output in progress_bar:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        logits = model(src, tgt_input)
        
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate_epoch(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in progress_bar:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            logits = model(src, tgt_input)
            
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt_output.reshape(-1)
            )
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)