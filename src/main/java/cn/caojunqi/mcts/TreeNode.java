package cn.caojunqi.mcts;

import cn.caojunqi.common.Tuple;
import org.apache.commons.lang3.Validate;

import java.util.HashMap;
import java.util.Map;

/**
 * 树节点
 *
 * @author Caojunqi
 * @date 2021-12-23 11:58
 */
public class TreeNode {
    /**
     * 父节点
     */
    private TreeNode parent;
    /**
     * 子节点 <行为索引, 对应的节点>
     */
    private Map<Integer, TreeNode> children;
    /**
     * 当前节点访问次数
     */
    private int visits;
    private float q;
    private float u;
    private float priorProbability;

    public TreeNode(TreeNode parent, float priorProbability) {
        this.parent = parent;
        this.children = new HashMap<>();
        this.visits = 0;
        this.q = 0;
        this.u = 0;
        this.priorProbability = priorProbability;
    }

    /**
     * Expand tree by creating new child.
     *
     * @param actionProbs <动作索引,动作概率>
     */
    public void expand(Map<Integer, Float> actionProbs) {
        Validate.isTrue(actionProbs != null && !actionProbs.isEmpty());
        actionProbs.forEach((action, prob) -> {
            if (this.children.containsKey(action)) {
                return;
            }
            this.children.put(action, new TreeNode(this, prob));
        });
    }

    public Tuple<Integer, TreeNode> select() {
        float maxValue = Float.NEGATIVE_INFINITY;
        Map.Entry<Integer, TreeNode> bestChild = null;
        for (Map.Entry<Integer, TreeNode> child : children.entrySet()) {
            float childValue = child.getValue().getValue();
            if (childValue > maxValue) {
                maxValue = childValue;
                bestChild = child;
            }
        }
        Validate.notNull(bestChild);
        return new Tuple<>(bestChild.getKey(), bestChild.getValue());
    }

    /**
     * Update node values from leaf evaluation.
     *
     * @param leafValue the value of subtree evaluation from the current player's
     *                  perspective.
     */
    public void update(float leafValue) {
        // Count visit
        this.visits += 1;
        // Update q, a running average of values for all visits.
        this.q += 1.0 * (leafValue - this.q) / this.visits;
    }

    /**
     * Like a call to update(), but applied recursively for all ancestors.
     *
     * @param leafValue
     */
    public void updateRecursive(float leafValue) {
        // If it is not root, this node's parent should be updated first.
        if (!isRoot()) {
            this.parent.updateRecursive(-leafValue);
        }
        update(leafValue);
    }

    /**
     * Calculate and return the value for this node.
     * It is a combination of leaf evaluations Q, and this node's prior
     * adjusted for its visit count, u.
     *
     * @return
     */
    public float getValue() {
        this.u = MctsParameter.C_PUCT * this.priorProbability * (float) Math.sqrt(this.parent.visits) / (1 + this.visits);
        return this.q + this.u;
    }

    /**
     * Check if leaf node (i.e. no nodes below this haven been expanded).
     *
     * @return
     */
    public boolean isLeaf() {
        return this.children.isEmpty();
    }

    public boolean isRoot() {
        return this.parent == null;
    }

    public void setParent(TreeNode parent) {
        this.parent = parent;
    }

    public Map<Integer, TreeNode> getChildren() {
        return children;
    }

    public int getVisits() {
        return visits;
    }
}
