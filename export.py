import torch
import agent

batch_size = 1
input_shape = (3, 256, 512)
dummy_input = torch.randn(batch_size, *input_shape, device='cuda')
lane_agent = agent.Agent()
lane_agent.load_weights(804, "tensor(0.5786)")
lane_agent.cuda()
lane_agent.evaluate_mode()
lane_agent.export_tensorrt(dummy_input, "pinet.engine")
