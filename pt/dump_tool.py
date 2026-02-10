import torch

golden = torch.load('./q_golden.pt')
bit = torch.load('./q_bit.pt')

assert(torch.allclose(bit, golden, rtol=1e-4, atol=1e-4))
print("===== Total equal before mask")

golden_mask = torch.load('./q_golden_mask.pt')
bit_mask = torch.load('./q_bit_mask.pt')

for i in range(0, 64, 8):
    offset_st = i
    offset_ed = i + 8
    
    res = torch.allclose(
        golden_mask[offset_st:offset_ed, offset_st:offset_ed], 
        bit_mask[offset_st:offset_ed, offset_st:offset_ed],
        rtol=1e-3,
        atol=1e-3
    )
    print(f"offset: {offset_st} - {offset_ed} {res}")