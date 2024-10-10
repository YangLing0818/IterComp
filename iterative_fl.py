import os

from Iterative_feedback import feedback_train

if __name__ == "__main__":
    args = feedback_train.parse_args()
    trainer = feedback_train.Trainer("stabilityai/stable-diffusion-xl-base-1.0", "data/itercomp_train_data.json", args=args)
    trainer.train(args=args)