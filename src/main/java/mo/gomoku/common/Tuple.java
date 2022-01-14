package mo.gomoku.common;

/**
 * 二元数据结构
 *
 * @author Caojunqi
 * @date 2021-11-22 16:42
 */
public class Tuple<First, Second> {
	public final First first;
	public final Second second;

	public Tuple(First first, Second second) {
		this.first = first;
		this.second = second;
	}
}
