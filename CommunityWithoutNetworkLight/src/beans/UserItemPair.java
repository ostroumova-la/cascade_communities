package beans;

import java.io.*;

public class UserItemPair implements Serializable {

	private static final long serialVersionUID = -6430075596101810618L;
	public int userId, itemId;

	public UserItemPair(int userId, int itemId) {
		this.userId = userId;
		this.itemId = itemId;
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
		UserItemPair other = (UserItemPair) obj;
		if (itemId != other.itemId)
			return false;
		if (userId != other.userId)
			return false;
		return true;
	}
}