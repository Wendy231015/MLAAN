Layer = {
    'resnet34': {
        1: [[4,2]],  # End-to-end
        16: [[1,0],[1,1],[1,2],
             [2,0],[2,1],[2,2],[2,3],
             [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],
             [4,0],[4,1],[4,2]],
    },
    'resnet101': {
        1:  [[4,2]],  # End-to-end
        33: [[1,0],[1,1],[1,2],
             [2,0],[2,1],[2,2],[2,3],
             [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
             [3,11],[3,12],[3,13],[3,14],[3,15],[3,16],[3,17],[3,18],[3,19],[3,20],[3,21],[3,22],
             [4,0],[4,1],[4,2]],
    }
}