package beans;

import java.io.*;

public class Activation implements Comparable<Activation>, Serializable {

	private static final long serialVersionUID = 4234048025528902299L;

	public int userId, itemId;
	public long timeStamp;

	public Activation(int userId, int itemId) {
		this.userId = userId;
		this.itemId = itemId;
	}

	public Activation(int userId, int itemId, long timeStamp) {
		this(userId, itemId);
		this.timeStamp = timeStamp;
	}

	public Activation getCopy() {
		return new Activation(userId, itemId);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + itemId;
		result = prime * result + userId;
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
		Activation other = (Activation) obj;
		if (itemId != other.itemId)
			return false;
		if (userId != other.userId)
			return false;
		return true;
	}

	@Override
	public int compareTo(Activation a) {
		if (timeStamp < a.timeStamp)
			return -1;
		if (timeStamp > a.timeStamp)
			return 1;
		if (userId < a.userId)
			return -1;
		if (userId > a.userId)
			return 1;
		if (itemId < a.itemId)
			return -1;
		if (itemId > a.itemId)
			return 1;
		return 0;
	}

	@Override
	public String toString() {
		return userId + "\t" + itemId + "\t" + timeStamp;
	}
}