package cn.caojunqi;

import org.apache.commons.collections4.queue.CircularFifoQueue;

/**
 * 队列测试类
 *
 * @author Caojunqi
 * @date 2022-01-12 17:19
 */
public class CircularFifoQueueTest {
    public static void main(String[] args) {
        CircularFifoQueue<Integer> datas = new CircularFifoQueue<>(3);
        datas.add(1);
        System.out.println(datas.size());
        System.out.println(datas);
        datas.add(2);
        System.out.println(datas.size());
        System.out.println(datas);
        datas.add(3);
        System.out.println(datas.size());
        System.out.println(datas);
        datas.add(4);
        System.out.println(datas.size());
        System.out.println(datas);
    }
}
