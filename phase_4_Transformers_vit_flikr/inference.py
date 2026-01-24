import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_caption_beam(
    image,
    encoder,
    decoder,
    vocab,
    device,
    beam_size=5,
    max_len=30,
    length_penalty=0.7
):
    encoder.eval()
    decoder.eval()

    bos = vocab.word2idx["<start>"]
    eos = vocab.word2idx["<end>"]

    image = image.unsqueeze(0).to(device)
    memory = encoder(image)

    beams = [(torch.tensor([[bos]], device=device), 0.0)]

    for _ in range(max_len):
        candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == eos:
                candidates.append((seq, score))
                continue

            logits = decoder(memory, seq)
            log_probs = F.log_softmax(logits[:, -1], dim=-1)

            topk_logp, topk_idx = log_probs.topk(beam_size)

            for k in range(beam_size):
                next_seq = torch.cat(
                    [seq, topk_idx[:, k].unsqueeze(1)], dim=1
                )
                new_score = score + topk_logp[0, k].item()
                candidates.append((next_seq, new_score))

        beams = sorted(
            candidates,
            key=lambda x: x[1] / (len(x[0][0]) ** length_penalty),
            reverse=True
        )[:beam_size]

    best = beams[0][0].squeeze(0).tolist()

    caption = []
    for idx in best:
        word = vocab.idx2word[idx]
        if word in ("<start>", "<pad>"):
            continue
        if word == "<end>":
            break
        caption.append(word)

    return " ".join(caption)
