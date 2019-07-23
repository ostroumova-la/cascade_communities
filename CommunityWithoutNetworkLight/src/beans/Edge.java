package beans;

import java.io.*;

public class Edge implements Serializable {

	private static final long serialVersionUID = 1L;

	public int source, destination;

	public Edge(int userId, int friendId) {
		this.source = userId;
		this.destination = friendId;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + destination;
		result = prime * result + source;
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
		Edge other = (Edge) obj;
		if (destination != other.destination)
			return false;
		if (source != other.source)
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "(" + source + ", " + destination + ")";
	}
}