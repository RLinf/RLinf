from rlinf.scheduler import Cluster, Worker


class MyWorker(Worker):
    def __init__(self):
        super().__init__()

    def run(self):
        import torch

        dim = 32 * 1024
        A = torch.randn(dim, dim).cuda()
        B = torch.randn(dim, dim).cuda()
        while True:
            C = torch.matmul(A, B)
            A = C


cluster = Cluster(num_nodes=1)
worker = MyWorker.create_group().launch(cluster)
worker.run().wait()
