import sys

# Programme de calcul externe auxiliaire pour l'Exemple 2
def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    outfile = "output.txt"
    
    params = {}
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                params[k.strip()] = float(v.strip())
    
    L = params.get("L", 2.5)
    F = params.get("F", 3000.0)
    E = params.get("E", 2.1e11)
    I = params.get("I", 4.0e-6)
    
    f_val = (F * L**3) / (3.0 * E * I)
    
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"f={f_val}\n")

if __name__ == "__main__":
    main()
