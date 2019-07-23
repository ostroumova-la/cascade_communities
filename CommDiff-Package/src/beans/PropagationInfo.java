package beans;

import java.io.*;

public class PropagationInfo implements Serializable {

    private static final long serialVersionUID = 5844688319563479307L;

    public int userId, community;
    public double eta;

    public PropagationInfo(int userId, int community, double eta) {
        this.userId = userId;
        this.community = community;
        this.eta = eta;
    }

    @Override
    public String toString() {
        return "PropagationInfo [userId=" + userId + ", community=" + community
                + ", eta=" + eta + "]";
    }
}