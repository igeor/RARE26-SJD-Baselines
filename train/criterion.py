import torch
from torch import nn
from typing import List
from torch.nn import functional as F

class DINOCenteredLoss(nn.Module):
    """DINO-style multi-crop loss with a running teacher center."""

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def set_teacher_temp(self, teacher_temp: float) -> None:
        self.teacher_temp = teacher_temp

    def forward(self, student_outputs: List[torch.Tensor], teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        student_outputs: outputs for all crops [global1, global2, local...]
        teacher_outputs: outputs for global crops only [global1, global2]
        """
        student_log_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs]
        teacher_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_outputs]

        total_loss = 0.0
        n_terms = 0
        for teacher_idx, tq in enumerate(teacher_probs):
            for student_idx, slogp in enumerate(student_log_probs):
                # Skip matching the same global view to itself.
                if student_idx == teacher_idx:
                    continue
                loss = torch.sum(-tq * slogp, dim=-1).mean()
                total_loss = total_loss + loss
                n_terms += 1

        if n_terms == 0:
            raise RuntimeError("No DINO loss terms were created. Check number of crops.")

        self.update_center(teacher_outputs)
        return total_loss / n_terms

    @torch.no_grad()
    def update_center(self, teacher_outputs: List[torch.Tensor]) -> None:
        teacher_concat = torch.cat(teacher_outputs, dim=0)
        batch_center = teacher_concat.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)

