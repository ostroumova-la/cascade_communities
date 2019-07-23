package beans;

import it.unimi.dsi.fastutil.ints.*;

import java.io.*;

public class FPair implements Serializable {

    private static final long serialVersionUID = -5006237833100609126L;

    public IntArrayList fplus, fminus;

    public FPair(IntArrayList fplus, IntArrayList fmins) {
        super();
        this.fplus = fplus;
        this.fminus = fmins;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((fminus == null) ? 0 : fminus.hashCode());
        result = prime * result + ((fplus == null) ? 0 : fplus.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        FPair other = (FPair) obj;
        if (fminus == null) {
            if (other.fminus != null)
                return false;
        } else if (!fminus.equals(other.fminus))
            return false;
        if (fplus == null) {
            if (other.fplus != null)
                return false;
        } else if (!fplus.equals(other.fplus))
            return false;
        return true;
    }
}