package mo.gomoku.common;

/**
 * 三元数据结构
 *
 * @author Caojunqi
 * @date 2022-01-12 17:03
 */
public class Triple<First, Second, Third> {
	public final First first;
	public final Second second;
	public final Third third;

	public Triple(First first, Second second, Third third) {
		this.first = first;
		this.second = second;
		this.third = third;
	}
}
