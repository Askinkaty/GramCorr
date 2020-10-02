BEGIN {
    fp=tp=fn=tn=0

    error_ids_seen[0] = ""
    error_ids_ignore[0] = ""
    prime = 1
}

{
    errorid = $1
    gold = $2
    pred = $3
    conf = $4

    if ($1 in error_ids_ignore) {
        print "woopsi!";
        ;
    } else if (error_ids_seen[errorid] < prime) {
        error_ids_seen[errorid] += 1

        if ((pred == 1) && (gold == 0)) {
            fp += 1
        } else if ((pred == 1) && (gold == 1)) {
            tp += 1
        } else if ((pred == "-1") && (gold == 0)) {
            tn += 1
        } else if ((pred == "-1") && (gold == 1)) {
            fn += 1
        } else {
            ;
        }
    } else {
        ;
    }
}

END {
    print fp,tp,fn,tn,fp+tp+fn+tn
    print tp/(tp+fp)
    print (tp+tn)/(tp+tn+fp+fn)
}
