import numpy as np
import struct
import sys
import os
from sklearn.neighbors import NearestNeighbors

def get_lid(data, k=20):
    nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(data)
    dists, _ = nbrs.kneighbors(data)
    dists = np.maximum(dists[:, 1:], 1e-10)
    return (k-1) / np.sum(np.log(dists[:, -1][:, None] / dists[:, :-1]), axis=1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calc_lid.py <base.bin>")
        return
        
    path = sys.argv[1]
    print(f"Processing {path}...")
    
    with open(path, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, d)
        
    lid = get_lid(data)
    lid = np.clip(lid, 0.1, 200.0) # Clip outliers
    
    out_path = path.replace("_base.bin", "_lid.bin")
    print(f"Saving to {out_path}, Mean LID: {np.mean(lid):.4f}")
    
    with open(out_path, "wb") as f:
        f.write(struct.pack("i", len(lid)))
        f.write(struct.pack("i", 1))
        f.write(lid.astype(np.float32).tobytes())

if __name__ == "__main__":
    main()