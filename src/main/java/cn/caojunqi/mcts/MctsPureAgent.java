package cn.caojunqi.mcts;

import cn.caojunqi.common.Tuple;
import cn.caojunqi.game.Board;
import org.apache.commons.lang3.Validate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/**
 * @author Caojunqi
 * @date 2022-01-12 21:41
 */
public class MctsPureAgent implements IAgent {

    private MctsPureCore core;

    public MctsPureAgent(Random random, int playout) {
        Function<Board, Tuple<Map<Integer, Float>, Float>> policyValueFn = board -> {
            List<Integer> availables = board.getAvailables();
            float prob = 1 / (float) availables.size();
            Map<Integer, Float> actionProbs = new HashMap<>(availables.size());
            for (int available : availables) {
                actionProbs.put(available, prob);
            }
            return new Tuple<>(actionProbs, 0f);
        };
        this.core = new MctsPureCore(random, policyValueFn, playout);
    }

    @Override
    public int getAction(Board board) {
        List<Integer> availables = board.getAvailables();
        Validate.isTrue(!availables.isEmpty());
        int move = this.core.getMove(board);
        this.core.updateWithMove(-1);
        return move;
    }
}
