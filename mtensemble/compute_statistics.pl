#!/usr/bin/perl
#
#
### Quick hack to run this evaluation script on the '6-files' dataset:
#
# for FILE in fold-?_01-o?_?-rf.pred
# do 
#   echo $FILE
#   awk 'BEGIN { FS=",";OFS="\t" } NF {gsub("^[21]:","",$2); gsub("^[21]:","",$3); print $6,"_error_type_","_org_length_","_correction_",$2,$3,-$5}' $FILE | ./compute_statistics.pl
# done
#

use strict;
use warnings;


# only one argument on the command line => the number of errors for which no system at all gave an answer
my ($nb_no_answers) = @ARGV;

my $results = {};

#getting the headers
my $first_line = <STDIN>;
chomp($first_line);

my ($error_id_label,$error_type_label,$original_length, $correction_label, $correct_incorrect_label, @rest_first_line) =  split (/\t/,$first_line);

#registering the names of the different systems
my @names=();
while(scalar(@rest_first_line) != 0){
    my ($opinion_label, $confidence_label,@other_rest_first_line) = @rest_first_line;
    @rest_first_line = @other_rest_first_line;
    

    if((!defined $opinion_label) || (!defined $confidence_label)){
	print STDERR "Something is wrong with the number of columns for the header at component ".(scalar(@names)+1)."\n";
	exit(1);
    }
    push(@names,$confidence_label);
}
if(scalar(@names) == 0){
    print STDERR "Something is wrong with the hearder since there are no column for components\n";
    exit(1);
}


#the overall number of error-id = the error-ids that got no answers (number provided on command line) + the error-ids that got an answer (added below)
my $nb_questions = $nb_no_answers;


#dealing with the other lines
my $line_number = 1;
while(my $line = <STDIN>){
    chomp($line);

    #taking the first four columns and leaving the rest n '@rest'
    my ($error_id,$error_type,$original_length, $correction, $correct_incorrect, @rest) = split (/\t/,$line);

    
    if(!exists $results->{$error_id}){
	$results->{$error_id} = {};
	#the number of questions with an answer increases
	$nb_questions++;
    }
    $line_number++;

    # basic check on the corrrect_incorrect value
    if(($correct_incorrect != 1) && ($correct_incorrect != 0)){
	print STDERR "Something is wrong with the correct_incorrect value '$correct_incorrect' at line $line_number\n";
	exit(1);
    }


    #copying the names of the components
    my @local_names = (@names);
    
    #now one system at the time
    while(scalar(@rest) != 0){
	my ($opinion,$confidence,@other_rest) = @rest;
	@rest = @other_rest;
	
	my $component_name = shift @local_names;

	if((!defined $opinion) || (!defined $confidence)){
	    print STDERR "Something is wrong with the number of column for component '$component_name' at line $line_number\n";
	    exit(1);
	}
    
        if(($opinion == -1) || ($opinion == 0)){
	    #doesn't have an opinion so nothing to register
	    next;
	}elsif($opinion != 1){
	    #just a check 
	    print STDERR "Something is wrong with the opinion value '$opinion'\n";
	    exit(1);
	}else{
	    #the system did generate this answer with a certain confidence score
	    push(@{$results->{$error_id}->{$component_name}},[$confidence,$correct_incorrect]);
	}
    }

    if(scalar(@local_names) != 0){
	 print STDERR "Something is wrong with the number of columns on line '$line_number', the should have been more of them\n";
	 exit(1);
    }
	
}

# the different hashes to register each component stats 
my ($nb_correct,$nb_correct_p,$nb_correct_p_p) = ({},{},{});
my $nb_questions_answered = {};

foreach my $error_id (keys %$results){
    #if a component gave an answer for this error-id then there will be an entry for it in the hash
    foreach my $component_name (keys %{$results->{$error_id}}){
	
	#ranking the answers for this error_id in decreasing order (higher score first)
	my @answers = sort {$b->[0] <=> $a->[0]} @{$results->{$error_id}->{$component_name}};

	# this component answered one more error-id	
	$nb_questions_answered->{$component_name}++;
	
	my ($first_answer,$second_answer,$third_answer) = @answers;
	$nb_correct->{$component_name} += $first_answer->[1];

	#just a check
	if($first_answer->[1] > 1){
	    print STDERR "WTF ?? 1 \n"; exit(1);
	}
        
	if(defined $second_answer){
	    $nb_correct_p->{$component_name} += $second_answer->[1];
	    
	    #just a check
	    if(($first_answer->[1] +  $second_answer->[1]) > 1){
		print STDERR "WTF ?? 2 \n"; exit(1);
	    }
	
	    if(defined $third_answer){
		$nb_correct_p_p->{$component_name} += $third_answer->[1];
	
		#just a check
		if(($first_answer->[1] +  $second_answer->[1] + $third_answer->[1]) > 1){
		    print STDERR "WTF ?? 3 \n"; exit(1);
		}
	    }
	}
    }
}

print "\n\n";

foreach my $component_name (@names){
    #adding the correct answers for the "only the first answer" set to the "only the two first answers" set
    $nb_correct_p->{$component_name} += $nb_correct->{$component_name};

    #adding the correct answers for the "only the first two answers" set to the "only the three first answers" set
    $nb_correct_p_p->{$component_name} += $nb_correct_p->{$component_name};

    print "*** $component_name ***\n"; 
    print "Precision   ".sprintf("%.3f", $nb_correct->{$component_name}*100/$nb_questions_answered->{$component_name})." ($nb_correct->{$component_name}/$nb_questions_answered->{$component_name}) \t\t "
	."Accuracy   ".sprintf("%.3f",$nb_correct->{$component_name}*100/$nb_questions)." ($nb_correct->{$component_name}/$nb_questions)\n";
    
    print "Precision*  ".sprintf("%.3f", $nb_correct_p->{$component_name}*100/$nb_questions_answered->{$component_name})." ($nb_correct_p->{$component_name}/$nb_questions_answered->{$component_name}) \t\t "
	."Accuracy*  ".sprintf("%.3f",$nb_correct_p->{$component_name}*100/$nb_questions)." ($nb_correct_p->{$component_name}/$nb_questions)\n";
    
    print "Precision** ".sprintf("%.3f", $nb_correct_p_p->{$component_name}*100/$nb_questions_answered->{$component_name})." ($nb_correct_p_p->{$component_name}/$nb_questions_answered->{$component_name})\t\t "
	."Accuracy**   ".sprintf("%.3f",$nb_correct_p_p->{$component_name}*100/$nb_questions)." ($nb_correct_p_p->{$component_name}/$nb_questions)\n";
    print "\n\n";
}


